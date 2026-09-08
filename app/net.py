import asyncio
import ipaddress
import logging
import socket
import time
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)

MAX_HOPS = 5
REDIRECT_CHAIN_BUDGET = 10.0


class UnsafeURL(Exception):
    """The URL is off the allow-list, non-http(s), or resolves somewhere private."""


def host_allowed(host: str, allowed: frozenset[str]) -> bool:
    host = host.lower().removeprefix("www.")
    return any(host == a or host.endswith("." + a) for a in allowed)


def _resolved_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURL(f"{host} does not resolve") from exc
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
        or any(ip in net for net in BLOCKED_NETWORKS)
    )


def assert_safe_url(url: str, allowed: frozenset[str]) -> None:
    """Raises UnsafeURL unless the URL is http(s), allow-listed, and publicly routable."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURL(f"scheme {parts.scheme!r} is not http(s)")

    host = parts.hostname
    if not host:
        raise UnsafeURL("url has no host")
    if not host_allowed(host, allowed):
        raise UnsafeURL(f"{host} is not in ALLOWED_DOMAINS")

    for ip in _resolved_ips(host):
        if is_blocked_ip(ip):
            raise UnsafeURL(f"{host} resolves to non-public address {ip}")


async def safe_get(
    url: str,
    *,
    allowed: frozenset[str],
    headers: dict[str, str],
    timeout: float,
    disable_redirects: bool,
) -> httpx.Response:
    """GET with the allow-list and private-IP check applied to every hop.

    httpx never follows a redirect itself. Each hop is re-validated here, and the whole
    chain shares one REDIRECT_CHAIN_BUDGET so a redirect loop cannot outlive it.
    """
    deadline = time.monotonic() + REDIRECT_CHAIN_BUDGET
    current = url

    async with httpx.AsyncClient(
        timeout=timeout, headers=headers, follow_redirects=False
    ) as client:
        for hop in range(MAX_HOPS):
            assert_safe_url(current, allowed)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UnsafeURL(f"redirect chain exceeded {REDIRECT_CHAIN_BUDGET}s")

            resp = await asyncio.wait_for(client.get(current), timeout=remaining)

            if not resp.is_redirect:
                return resp
            if disable_redirects:
                raise UnsafeURL(f"redirect refused (DISABLE_REDIRECTS): {current}")

            nxt = resp.next_request
            if nxt is None:
                return resp
            log.info("redirect hop %d: %s -> %s", hop + 1, current, nxt.url)
            current = str(nxt.url)

    raise UnsafeURL(f"more than {MAX_HOPS} redirects from {url}")
