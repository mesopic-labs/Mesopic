"""The engine's exception hierarchy.

One root so a caller can catch everything Muster raises without catching the world, and
one class per *recoverable situation the pipeline is designed around* — a dropped stream
is a routine event with a defined response (§4), not an exception in the pejorative sense.
"""

from __future__ import annotations


class MusterError(Exception):
    """Root of every error the engine raises deliberately."""


class ConfigError(MusterError):
    """``muster.yaml`` is invalid: bad types, out-of-range values, dangling references.

    Raised at startup. A bad config must fail loud rather than silently mis-count
    (engine-architecture.md §13.1).
    """


class StreamDropped(MusterError):  # noqa: N818 - the name engine-architecture.md §4 uses
    """The camera stream ended or stalled. Ingest reconnects with backoff (§4)."""


class ModelError(MusterError):
    """The detector could not fetch, export, quantize, or load a model (§6)."""


class StoreError(MusterError):
    """The SQLite store could not complete an operation (§11)."""


class ExportError(MusterError):
    """An exporter failed to deliver. Exporters fail independently — never a stall (§12)."""


class SyncError(MusterError):
    """The metrics-sync client could not ship a batch. Rows stay unsynced and retry (§14)."""
