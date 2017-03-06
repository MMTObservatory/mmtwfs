# Licensed under GPL3


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