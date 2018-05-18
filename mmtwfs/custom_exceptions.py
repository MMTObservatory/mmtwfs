# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

__all__ = ['MMTWFSException', 'WFSConfigException', 'WFSCommandException', 'WFSAnalysisFailed', 'ZernikeException']


class MMTWFSException(Exception):
    """
    Superclass of all custom exceptions

    This exception contains these fields of interest:

        message - the message provided by the code that raised the exception

        results - the results dict that can contain relevant data that can be reported at time of exception

    """

    def __init__(self, value='Unspecified Error', results=None):
        super(MMTWFSException, self).__init__(self, value)
        self.results = results
        self.name = "Unspecified mmtwfs exception"


class WFSConfigException(MMTWFSException):
    """
    Raise when an error occurs due to invalid configuration data.
    """
    def __init__(self, value="Config Error", results=None):
        super(WFSConfigException, self).__init__(value, results=results)
        self.name = "Config Error"


class WFSCommandException(MMTWFSException):
    """
    Raise when an error occurs due to invalid command sent to a WFS system.
    """
    def __init__(self, value="Command Error", results=None):
        super(WFSCommandException, self).__init__(value, results=results)
        self.name = "Command Error"


class WFSAnalysisFailed(MMTWFSException):
    """
    Raise when something is wrong with the WFS data that prevents it from being analyzed
    """
    def __init__(self, value="WFS Image Analysis Error", results=None):
        super(WFSAnalysisFailed, self).__init__(value, results=results)
        self.name = "Analysis Error"


class ZernikeException(MMTWFSException):
    """
    Raise when an error occurs in handling or configuring of ZernikeVectors
    """
    def __init__(self, value="Zernike Handling Error", results=None):
        super(ZernikeException, self).__init__(value, results=results)
        self.name = "Zernike Error"
