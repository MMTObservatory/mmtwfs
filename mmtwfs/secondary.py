# Licensed under GPL3
# coding=utf-8

"""
Classes and utilities for optical modeling and controlling the position of the secondary mirrors of the MMTO.
"""

from .config import recursive_subclasses, merge_config, mmt_config
from .custom_exceptions import WFSConfigException


def SecondaryFactory(secondary="f5", config={}, **kwargs):
    """
    Build and return proper Secondary sub-class instance based on the value of 'secondary'.
    """
    config = merge_config(config, dict(**kwargs))
    secondary = secondary.lower()

    types = recursive_subclasses(Secondary)
    secondaries = [t.__name__.lower() for t in types]
    sec_map = dict(list(zip(secondaries, types)))

    if secondary not in secondaries:
        raise WFSConfigException(value="Specified secondary, %s, not valid or not implemented." % secondary)

    sec_cls = sec_map[secondary](config=config)
    return sec_cls


class Secondary(object):
    def __init__(self, config={}):
        key = self.__class__.__name__.lower()
        self.__dict__.update(merge_config(mmt_config['secondary'][key], config))


class F5(Secondary):
    pass


class F9(Secondary):
    pass
