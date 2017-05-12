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

try:
    import tornado
except ImportError:
    raise RuntimeError("This example requires tornado.")
import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.websocket

import matplotlib.pyplot as plt
from matplotlib.backends.backend_webagg_core import (FigureManagerWebAgg, FigureCanvasWebAggCore, new_figure_manager_given_figure)
from matplotlib.figure import Figure

import numpy as np

import json

from mmtwfs.wfs import WFSFactory
from mmtwfs.zernike import ZernikeVector
from mmtwfs.telescope import MMT


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
                    print("setting %s" % wfs)
                    self.application.wfs = self.application.wfs_systems[wfs]
            except:
                print("Must specify valid wfs: %s." % wfs)

    class WFSPageHandler(tornado.web.RequestHandler):
        def get(self):
            try:
                wfs = self.get_argument("wfs")
                if wfs in self.application.wfs_keys:
                    print("setting %s" % wfs)
                    #self.application.refresh_figures()
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

                    self.render("wfs.html", wfsname=self.application.wfs.name, ws_uris=ws_uris, fig_ids=fig_ids, figures=figkeys)
            except Exception as e:
                print("Must specify valid wfs: %s. %s" % (wfs, e))

    class ConnectHandler(tornado.web.RequestHandler):
        def get(self):
            if self.application.wfs is not None:
                if not self.application.wfs.connected:
                    self.application.wfs.connect()

    class DisconnectHandler(tornado.web.RequestHandler):
        def get(self):
            if self.application.wfs is not None:
                if self.application.wfs.connected:
                    self.application.wfs.disconnect()

    class AnalyzeHandler(tornado.web.RequestHandler):
        def get(self):
            self.application.close_figures()
            try:
                filename = self.get_argument("fitsfile")
                print(filename)
            except:
                print("no wfs or file specified.")
            if os.path.isfile(filename):
                results = self.application.wfs.measure_slopes(filename, plot=True)
                zresults = self.application.wfs.fit_wavefront(results, plot=True)
                zvec = zresults['zernike']
                tel = self.application.wfs.telescope
                m1gain = self.application.wfs.m1_gain
                m2gain = self.application.wfs.m2_gain
                forces, m1focus = tel.calculate_primary_corrections(zvec, gain=m1gain)
                figures = {}
                figures['slopes'] = results['figures']['slopes']
                figures['residuals'] = zresults['resid_plot']
                figures['wavefront'] = zvec.plot_map()
                figures['barchart'] = zvec.bar_chart(residual=zresults['residual_rms'])
                figures['forces'] = tel.plot_forces(forces, m1focus)
                figures['forces'].set_label("Requested M1 Actuator Forces")
                figures['totalforces'] = tel.plot_forces(tel.total_forces)
                figures['totalforces'].set_label("Total M1 Actuator Forces")
                psf, figures['psf'] = tel.psf(zv=zvec.copy())
                print(zvec)
                self.application.wavefront_fit = zvec
                self.application.pending_forces = forces
                self.application.pending_m1focus = m1focus
                self.application.pending_focus = self.application.wfs.calculate_focus(zvec)
                self.application.pending_cc_x, self.application.pending_cc_y = self.application.wfs.calculate_cc(zvec)

                figures['barchart'].axes[0].set_title("Focus: {0:0.1f}".format(self.application.pending_focus))

                self.application.refresh_figures(figures=figures)

    class M1CorrectHandler(tornado.web.RequestHandler):
        def get(self):
            print("m1correct")

    class M2CorrectHandler(tornado.web.RequestHandler):
        def get(self):
            print("m2correct")

    class RecenterHandler(tornado.web.RequestHandler):
        def get(self):
            print("recenter")

    class RestartHandler(tornado.web.RequestHandler):
        def get(self):
            try:
                wfs = self.get_argument('wfs')
                print("restarting %s" % wfs)
            except:
                print("no wfs specified")

    class DataDirHandler(tornado.web.RequestHandler):
        def get(self):
            try:
                datadir = self.get_argument("datadir")
                print(datadir)
            except:
                print("no datadir specified")

    class M1GainHandler(tornado.web.RequestHandler):
        def get(self):
            try:
                gain = float(self.get_argument(gain))
                for k, w in self.application.wfs_systems.items():
                    print("seeing m1_gain to %f in %s" % (gain, k))
            except:
                print("no m1_gain specified")

    class M2GainHandler(tornado.web.RequestHandler):
        def get(self):
            try:
                gain = float(self.get_argument(gain))
                for k, w in self.application.wfs_systems.items():
                    print("seeing m2_gain to %f in %s" % (gain, k))
            except:
                print("no m2_gain specified")

    class MplJs(tornado.web.RequestHandler):
        """
        Serves the generated matplotlib javascript file.  The content
        is dynamically generated based on which toolbar functions the
        user has defined.  Call `FigureManagerWebAgg` to get its
        content.
        """
        def get(self):
            self.set_header('Content-Type', 'application/javascript')

            js_content = FigureManagerWebAgg.get_javascript()

            self.write(js_content)

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
        self.wfs_systems[wfs] = wfs
        print("self.wfs_systems[wfs] = WFSFactory(wfs=wfs)")

    def close_figures(self):
        if self.figures is not None:
            for k, f in self.figures.items():
                plt.close(f)

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
                self.managers[k].refresh_all()

    def __init__(self):
        if 'WFSROOT' in os.environ:
            self.datadir = os.environ['WFSROOT']
        else:
            self.datadir = "/mmt/shwfs/datadir"

        self.wfs = None
        self.wfs_systems = {}
        self.wfs_keys = ['newf9', 'f9', 'f5', 'mmirs']
        self.wfs_names = {}
        for w in self.wfs_keys:
            self.wfs_systems[w] = WFSFactory(wfs=w)
            self.wfs_names[w] = self.wfs_systems[w].name

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
            (r'/mpl.js', self.MplJs),
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
