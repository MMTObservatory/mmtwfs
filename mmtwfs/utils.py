# Licensed under a 3-clause BSD style license - see LICENSE.rst
# coding=utf-8

"""
Misc. utility routines
"""

import dns.resolver as resolver

import logging
import logging.handlers

log = logging.getLogger("DNSresolver")
log.setLevel(logging.INFO)

__all__ = ["srvlookup"]


def srvlookup(server):
    """
    Perform a SRV lookup of 'server' and return its hostname and port.
    """
    try:
        response = resolver.resolve(server, "SRV")
        host = response[0].target.to_text()
        port = response[0].port
    except Exception as e:
        # we'll be lenient and just return None if there's an issue
        log.warning(f"Problem looking up {server} via DNS: {e}")
        host = None
        port = None

    return host, port
