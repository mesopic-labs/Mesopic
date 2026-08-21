"""The engine's exception hierarchy.

One root so a caller can catch everything Mesopic raises without catching the world, and
one class per *recoverable situation the pipeline is designed around* — a dropped stream
is a routine event with a defined response (§4), not an exception in the pejorative sense.
"""

from __future__ import annotations


class MesopicError(Exception):
    """Root of every error the engine raises deliberately."""


class ConfigError(MesopicError):
    """``mesopic.yaml`` is invalid: bad types, out-of-range values, dangling references.

    Raised at startup. A bad config must fail loud rather than silently mis-count
    (engine-architecture.md §13.1).
    """


class StreamDropped(MesopicError):  # noqa: N818 - the name engine-architecture.md §4 uses
    """The camera stream ended or stalled. Ingest reconnects with backoff (§4)."""


class ModelError(MesopicError):
    """The detector could not fetch, export, quantize, or load a model (§6)."""


class StoreError(MesopicError):
    """The SQLite store could not complete an operation (§11)."""


class ExportError(MesopicError):
    """An exporter failed to deliver. Exporters fail independently — never a stall (§12)."""


class SyncError(MesopicError):
    """The metrics-sync client could not ship a batch. Rows stay unsynced and retry (§14)."""


class TruthError(MesopicError):
    """A ground-truth artefact is unusable: malformed, inconsistent, or ineligible.

    Covers both halves of MK.2 — a truth file or clip manifest that fails validation, and
    the refusal to gate a release on footage whose provenance or consent does not permit
    it. Both are loud by design: a quietly-accepted bad label produces an accuracy number
    that looks fine and is wrong.
    """


class SnapshotUnavailableError(MesopicError):
    """No calibration frame could be got from a camera, and why.

    Refusing is the design (P3.8): the snapshot comes from the worker that already owns
    the stream, so a camera that is not streaming has none to give. The message names the
    camera's state so the operator fixes the stream rather than the drawing tool.
    """
