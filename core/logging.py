import logging
import sys

_NOISY_HTTP_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "hpack",
    "hpack.hpack",
    "hpack.table",
    "openai",
    "supabase",
    "postgrest",
    "gotrue",
    "realtime",
    "storage3",
    "supafunc",
)


def setup_logging(debug: bool = True) -> None:
    """Configure app logging without leaking secrets from HTTP client DEBUG traces."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    if debug:
        for name in ("domains", "infra", "api", "core", "main", "scripts"):
            logging.getLogger(name).setLevel(logging.DEBUG)

    for name in _NOISY_HTTP_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
