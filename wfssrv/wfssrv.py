"""
This example demonstrates how to embed matplotlib WebAgg interactive
plotting in your own web application and framework.  It is not
necessary to do all this if you merely want to display a plot in a
browser or use matplotlib's built-in Tornado-based server "on the
side".

The framework being used must support web sockets.
"""

import io
import os
import socket
import glob
import json

import logging
import logging.handlers
logger = logging.getLogger("")
logger.setLevel(logging.INFO)

try:
    import tornado
except ImportError:
    raise RuntimeError("This example requires tornado.")
import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.websocket
from tornado.log import enable_pretty_logging
enable_pretty_logging()

import matplotlib
matplotlib.use('webagg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_webagg_core import (FigureManagerWebAgg, FigureCanvasWebAggCore, new_figure_manager_given_figure)
from matplotlib.figure import Figure

import numpy as np

import astropy.units as u

from mmtwfs.wfs import WFSFactory
from mmtwfs.zernike import ZernikeVector
from mmtwfs.telescope import MMT

log = logging.getLogger('tornado.application')
log.setLevel(logging.INFO)

def create_default_figures():
    zv = ZernikeVector(Z04=1)
    figures = {}
    ax = {}
    data = np.zeros((512, 512))
    tel = MMT(secondary='f5')  # secondary doesn't matter, just need methods for mirror forces/plots
    forces = tel.bending_forces(zv=zv)

    # stub for plot showing bkg-subtracted WFS image with aperture positions
    figures['slopes'], ax['slopes'] = plt.subplots()
    figures['slopes'].set_label("Aperture Positions and Spot Movement")
    ax['slopes'].imshow(data, cmap='Greys', origin='lower', interpolation='None')

    # stub for plot showing bkg-subtracted WFS image and residuals slopes of wavefront fit
    figures['residuals'], ax['residuals'] = plt.subplots()
    figures['residuals'].set_label("Zernike Fit Residuals")
    ax['residuals'].imshow(data, cmap='Greys', origin='lower', interpolation='None')

    # stub for zernike wavefront map
    figures['wavefront'] = zv.plot_map()

    # stub for zernike bar chart
    figures['barchart'] = zv.bar_chart()

    # stubs for mirror forces
    figures['forces'] = tel.plot_forces(forces)
    figures['forces'].set_label("Requested M1 Actuator Forces")

    # stubs for mirror forces
    figures['totalforces'] = tel.plot_forces(forces)
    figures['totalforces'].set_label("Total M1 Actuator Forces")

    # stub for psf
    psf, figures['psf'] = tel.psf(zv=zv)

    return figures


class WFSServ(tornado.web.Application):
    class HomeHandler(tornado.web.RequestHandler):
        """
        Serves the main HTML page.
        """
        def get(self):
            self.render("home.html", current=self.application.wfs, wfslist=self.application.wfs_names)

    class SelectHandler(tornado.web.RequestHandler):
        def get(self):
            try:
                wfs = self.get_argument("wfs")
                render = self.get_argument("render", default=False)
                if wfs in self.application.wfs_keys:
                    log.info("setting %s" % wfs)
                    self.application.wfs = self.application.wfs_systems[wfs]
            except:
                log.warn("Must specify valid wfs: %s." % wfs)

    class WFSPageHandler(tornado.web.RequestHandler):
        def get(self):
            try:
                wfs = self.get_argument("wfs")
                if wfs in self.application.wfs_keys:
                    log.info("setting %s" % wfs)
                    self.application.wfs = self.application.wfs_systems[wfs]
                    figkeys = []
                    ws_uris = []
                    fig_ids = []
                    for k, f in self.application.figures.items():
                        manager = self.application.managers[k]
                        fig_ids.append(manager.num)
                        figkeys.append(k)
                        ws_uri = "ws://{req.host}/{figdiv}/ws".format(req=self.request, figdiv=k)
                        ws_uris.append(ws_uri)

                    self.render(
                        "wfs.html",
                        wfsname=self.application.wfs.name,
                        ws_uris=ws_uris,
                        fig_ids=fig_ids,
                        figures=figkeys,
                        datadir=self.application.datadir + "/"
                    )
            except Exception as e:
                log.warn("Must specify valid wfs: %s. %s" % (wfs, e))

    class ConnectHandler(tornado.web.RequestHandler):
        def get(self):
            if self.application.wfs is not None:
                if not self.application.wfs.connected:
                    self.application.wfs.connect()
                    if self.application.wfs.connected:
                        log.info("%s is connected." % self.application.wfs.name)
                    else:
                        log.warn("Couldn't connect to %s. Offline?" % self.application.wfs.name)
                else:
                    log.info("%s already connected" % self.application.wfs.name)

    class DisconnectHandler(tornado.web.RequestHandler):
        def get(self):
            if self.application.wfs is not None:
                if self.application.wfs.connected:
                    self.application.wfs.disconnect()
                    log.info("%s is disconnected." % self.application.wfs.name)
                else:
                    log.info("%s already disconnected" % self.application.wfs.name)

    class AnalyzeHandler(tornado.web.RequestHandler):
        def get(self):
            self.application.close_figures()
            try:
                filename = self.get_argument("fitsfile")
                log.info("Analyzing %s..." % filename)
            except:
                log.warn("no wfs or file specified.")

            if os.path.isfile(filename):
                self.application.wfs.connect()
                results = self.application.wfs.measure_slopes(filename, plot=True)
                self.application.update_seeing(results['seeing'])
                zresults = self.application.wfs.fit_wavefront(results, plot=True)
                zvec = zresults['zernike']
                tel = self.application.wfs.telescope
                m1gain = self.application.wfs.m1_gain
                m2gain = self.application.wfs.m2_gain
                # this is the total if we try to correct everything as fit
                totforces, totm1focus = tel.calculate_primary_corrections(zvec, gain=m1gain)
                figures = {}
                figures['slopes'] = results['figures']['slopes']
                figures['residuals'] = zresults['resid_plot']
                figures['wavefront'] = zvec.plot_map()
                figures['barchart'] = zvec.bar_chart(residual=zresults['residual_rms'])
                figures['totalforces'] = tel.plot_forces(totforces, totm1focus)
                figures['totalforces'].set_label("Total M1 Actuator Forces")
                psf, figures['psf'] = tel.psf(zv=zvec.copy())
                log.info(zvec)
                zvec_file = os.path.join(self.application.datadir, filename + ".zernike")
                zvec.save(filename=zvec_file)

                # check the RMS of the wavefront fit and only apply corrections if the fit is good enough.
                # M2 can be more lenient to take care of large amounts of focus or coma.
                if zresults['residual_rms'] < 800 * u.nm:
                    self.application.has_pending_m2 = True
                if zresults['residual_rms'] < 400 * u.nm:
                    self.application.has_pending_m1 = True
                self.application.has_pending_recenter = True

                self.application.wavefront_fit = zvec
                self.application.pending_focus = self.application.wfs.calculate_focus(zvec)
                self.application.pending_cc_x, self.application.pending_cc_y = self.application.wfs.calculate_cc(zvec)
                self.application.pending_az, self.application.pending_el = self.application.wfs.calculate_recenter(results)
                self.application.pending_forces, self.application.pending_m1focus = \
                    self.application.wfs.calculate_primary(zvec, threshold=m1gain*zresults['residual_rms'])
                self.application.pending_forcefile = os.path.join(self.application.datadir, filename + ".forces")
                limit = np.round(np.abs(self.application.pending_forces['force']).max())
                figures['forces'] = tel.plot_forces(
                    self.application.pending_forces,
                    self.application.pending_m1focus,
                    limit=limit
                )
                figures['forces'].set_label("Requested M1 Actuator Forces")
                figures['barchart'].axes[0].set_title("Focus: {0:0.1f}  CC_X: {1:0.1f}  CC_Y: {2:0.1f}".format(
                        self.application.pending_focus,
                        self.application.pending_cc_x,
                        self.application.pending_cc_y,
                    )
                )

                self.application.refresh_figures(figures=figures)

    class M1CorrectHandler(tornado.web.RequestHandler):
        def get(self):
            log.info("M1 corrections")
            if self.application.has_pending_m1 and self.application.wfs.connected:
                self.application.wfs.telescope.correct_primary(
                    self.application.pending_forces,
                    self.application.pending_m1focus,
                    filename=self.application.pending_forcefile
                )
                log.info(self.application.pending_forces)
                log.info("m1focus = %s" % self.application.pending_m1focus)
                self.application.has_pending_m1 = False
                self.write("Sending forces to cell and {0:0.1f} focus to secondary...".format(self.application.pending_m1focus)
            else:
                log.info("no M1 corrections sent")
                self.write("No M1 corrections sent")

    class M2CorrectHandler(tornado.web.RequestHandler):
        def get(self):
            log.info("M2 corrections")
            if self.application.has_pending_m2 and self.application.wfs.connected:
                self.application.wfs.secondary.focus(self.application.pending_focus)
                self.application.wfs.secondary.correct_coma(self.application.pending_cc_x, self.application.pending_cc_y)
                self.application.has_pending_m2 = False
                self.write("Sending {0:0.1f} focus and {1:0.1f}/{2:0.1f} CC_X/CC_Y to secondary...".format(
                    self.application.pending_focus,
                    self.application.pending_cc_x,
                    self.application.pending_cc_y
                ))
            else:
                log.info("no M2 corrections sent")
                self.write("No M2 corrections sent")

    class RecenterHandler(tornado.web.RequestHandler):
        def get(self):
            log.info("Recentering...")
            if self.application.has_pending_recenter and self.application.wfs.connected:
                self.application.wfs.secondary.recenter(self.application.pending_az, self.application.pending_el)
                self.application.has_pending_recenter = False
                self.write("Sending {0:0.1f}/{1:0.1f} of az/el to recenter...".format(
                    self.application.pending_az,
                    self.application.pending_el
                ))
            else:
                log.info("no M2 recenter corrections sent")
                self.write("No M2 recenter corrections sent")

    class RestartHandler(tornado.web.RequestHandler):
        def get(self):
            try:
                wfs = self.get_argument('wfs')
                self.application.restart_wfs(wfs)
                log.info("restarting %s" % wfs)
            except:
                log.info("no wfs specified")

    class DataDirHandler(tornado.web.RequestHandler):
        def get(self):
            try:
                datadir = self.get_argument("datadir")
                if os.path.isdir(datadir):
                    log.info("setting datadir to %s" % datadir)
                    self.application.datadir = datadir
            except:
                log.info("no datadir specified")

    class M1GainHandler(tornado.web.RequestHandler):
        def get(self):
            if self.application.wfs is not None:
                self.write("%f" % self.application.wfs.m1_gain)
            else:
                self.write("no WFS selected.")

        def post(self):
            try:
                gain = float(self.get_argument(gain))
                if self.application.wfs is not None:
                    self.application.wfs.m1_gain = gain
            except:
                log.info("no m1_gain specified")

    class M2GainHandler(tornado.web.RequestHandler):
        def get(self):
            if self.application.wfs is not None:
                self.write("%f" % self.application.wfs.m2_gain)
            else:
                self.write("no WFS selected.")

        def post(self):
            try:
                gain = float(self.get_argument(gain))
                if self.application.wfs is not None:
                    self.application.wfs.m2_gain = gain
            except:
                log.info("no m2_gain specified")

    class PendingHandler(tornado.web.RequestHandler):
        def get(self):
            self.write("M1: %s" % self.application.has_pending_m1)
            self.write("M2: %s" % self.application.has_pending_m2)
            self.write("recenter: %s" % self.application.has_pending_recenter)

        def post(self):
            self.application.has_pending_m1 = False
            self.application.has_pending_m2 = False
            self.application.has_pending_recenter = False

    class FilesHandler(tornado.web.RequestHandler):
        def get(self):
            fullfiles = glob.glob(os.path.join(self.application.datadir, "*.fits"))
            files = []
            for f in fullfiles:
                files.append(os.path.split(f)[1])
            self.write(json.dumps(files))

    class Download(tornado.web.RequestHandler):
        """
        Handles downloading of the figure in various file formats.
        """
        def get(self, fig, fmt):
            managers = self.application.managers

            mimetypes = {
                'ps': 'application/postscript',
                'eps': 'application/postscript',
                'pdf': 'application/pdf',
                'svg': 'image/svg+xml',
                'png': 'image/png',
                'jpeg': 'image/jpeg',
                'tif': 'image/tiff',
                'emf': 'application/emf'
            }

            self.set_header('Content-Type', mimetypes.get(fmt, 'binary'))

            buff = io.BytesIO()
            managers[fig].canvas.print_figure(buff, format=fmt)
            self.write(buff.getvalue())

    class WebSocket(tornado.websocket.WebSocketHandler):
        """
        A websocket for interactive communication between the plot in
        the browser and the server.

        In addition to the methods required by tornado, it is required to
        have two callback methods:

            - ``send_json(json_content)`` is called by matplotlib when
              it needs to send json to the browser.  `json_content` is
              a JSON tree (Python dictionary), and it is the responsibility
              of this implementation to encode it as a string to send over
              the socket.

            - ``send_binary(blob)`` is called to send binary image data
              to the browser.
        """
        supports_binary = True

        def open(self, figname):
            # Register the websocket with the FigureManager.
            self.figname = figname
            manager = self.application.managers[figname]
            manager.add_web_socket(self)
            if hasattr(self, 'set_nodelay'):
                self.set_nodelay(True)

        def on_close(self):
            # When the socket is closed, deregister the websocket with
            # the FigureManager.
            manager = self.application.managers[self.figname]
            manager.remove_web_socket(self)

        def on_message(self, message):
            # The 'supports_binary' message is relevant to the
            # websocket itself.  The other messages get passed along
            # to matplotlib as-is.

            # Every message has a "type" and a "figure_id".
            message = json.loads(message)
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']
            else:
                manager = self.application.fig_id_map[message['figure_id']]
                manager.handle_json(message)

        def send_json(self, content):
            self.write_message(json.dumps(content))

        def send_binary(self, blob):
            if self.supports_binary:
                self.write_message(blob, binary=True)
            else:
                data_uri = "data:image/png;base64,{0}".format(
                    blob.encode('base64').replace('\n', ''))
                self.write_message(data_uri)

    def restart_wfs(self, wfs):
        """
        If there's a configuration change, provide a way to reload the specified WFS
        """
        del self.wfs_systems[wfs]
        self.wfs_systems[wfs] = WFSFactory(wfs=wfs)

    def close_figures(self):
        if self.figures is not None:
            plt.close('all')

    def refresh_figures(self, figures=None):
        if figures is None:
            self.figures = create_default_figures()
        else:
            self.figures = figures

        for k, f in self.figures.items():
            if k not in self.managers:
                fignum = id(f)
                self.managers[k] = new_figure_manager_given_figure(fignum, f)
                self.fig_id_map[fignum] = self.managers[k]
            else:
                canvas = FigureCanvasWebAggCore(f)
                self.managers[k].canvas = canvas
                self.managers[k].canvas.manager = self.managers[k]
                self.managers[k]._get_toolbar(canvas)
                self.managers[k]._send_event("refresh")
                self.managers[k].canvas.draw()

    def update_seeing(self, seeing):
        try:
            seeing_server = ("hacksaw.mmto.org", 7666)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(seeing_server)
            cmd = "set wfs_seeing {0:0.2f}".format(seeing)
            sock.sendall(cmd.encode("utf8"))
            sock.close()
        except Exception as e:
            log.warn("Error connecting to hacksaw... : %s" % e)

    def __init__(self):
        if 'WFSROOT' in os.environ:
            self.datadir = os.environ['WFSROOT']
        else:
            self.datadir = "/mmt/shwfs/datadir"

        if os.path.isdir(self.datadir):
            logfile = os.path.join(self.datadir, "wfs.log")
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler = logging.handlers.WatchedFileHandler(logfile)
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            enable_pretty_logging()

        self.wfs = None
        self.wfs_systems = {}
        self.wfs_keys = ['newf9', 'f9', 'f5', 'mmirs']
        self.wfs_names = {}
        for w in self.wfs_keys:
            self.wfs_systems[w] = WFSFactory(wfs=w)
            self.wfs_names[w] = self.wfs_systems[w].name

        self.has_pending_m1 = False
        self.has_pending_m2 = False
        self.has_pending_recenter = False

        self.figures = None
        self.managers = {}
        self.fig_id_map = {}
        self.refresh_figures()
        self.wavefront_fit = ZernikeVector(Z04=1)

        handlers = [
            (r"/", self.HomeHandler),
            (r"/select", self.SelectHandler),
            (r"/wfspage", self.WFSPageHandler),
            (r"/connect", self.ConnectHandler),
            (r"/disconnect", self.DisconnectHandler),
            (r"/analyze", self.AnalyzeHandler),
            (r"/m1correct", self.M1CorrectHandler),
            (r"/m2correct", self.M2CorrectHandler),
            (r"/recenter", self.RecenterHandler),
            (r"/restart", self.RestartHandler),
            (r"/setdatadir", self.DataDirHandler),
            (r"/m1gain", self.M1GainHandler),
            (r"/m2gain", self.M2GainHandler),
            (r"/clearpending", self.PendingHandler),
            (r"/files", self.FilesHandler),
            (r'/download_([a-z]+).([a-z0-9.]+)', self.Download),
            (r'/([a-z0-9.]+)/ws', self.WebSocket)
        ]

        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            debug=True
        )
        super(WFSServ, self).__init__(handlers, **settings)


if __name__ == "__main__":
    application = WFSServ()

    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)

    print("http://127.0.0.1:8080/")
    print("Press Ctrl+C to quit")

    tornado.ioloop.IOLoop.instance().start()
