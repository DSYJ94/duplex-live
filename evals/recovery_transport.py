"""Actual provider HTTP transport; optional scoped loopback SSH proxy.

The loopback route supplies no provider credential to the GPU host. The Mac
proxy forwards the unchanged payload to the real Responses API. It is never a
response executor, oracle, tool implementation or model-output substitution.
"""
import re
from urllib.parse import urlsplit

ENDPOINT = 'https://api.openai.com/v1/responses'
PROXY_ENV = 'DUPLEX_OPENAI_LOOPBACK_PROXY'
RUN_ENV = 'DUPLEX_OPENAI_PROXY_RUN'


def endpoint(environment):
    proxy = environment.get(PROXY_ENV, '')
    if proxy:
        value = urlsplit(proxy)
        try: port = value.port
        except ValueError: raise ValueError('Invalid loopback proxy port') from None
        run = environment.get(RUN_ENV, '')
        if (value.scheme != 'http' or value.hostname != '127.0.0.1'
                or not port or value.netloc != '127.0.0.1:' + str(port)
                or value.path != '/v1/responses' or value.query or value.fragment
                or not re.fullmatch(r'[a-zA-Z0-9_-]{1,96}', run)):
            raise ValueError('Only an explicitly scoped127.0.0.1 Responses proxy is allowed')
        # Deliberately ignore a provider key even if accidentally inherited.
        return proxy, {'X-Duplex-Run': run}, 'mac-loopback-responses-proxy'
    key = environment.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('Backend API credential unavailable')
    return ENDPOINT, {'Authorization': 'Bearer ' + key}, 'direct-openai-https'
