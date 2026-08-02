

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlparse


def url_host(url: str) -> str:
    try:
        host = (urlparse(url).netloc or "").lower()
    except ValueError:
        return ""
    return host.split("@")[-1].split(":")[0]


def host_in_domains(url: str, domains: Iterable[str]) -> bool:
    """True when the URL host equals or is a subdomain of an allowed domain."""
    host = url_host(url)
    if not host:
        return False
    return any(
        host == d or host.endswith("." + d)
        for d in (x.strip().lower() for x in domains)
        if d
    )
