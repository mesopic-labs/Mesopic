"""Spawn, watch, and restart camera workers; own the shared sinks.

The bounded event queue is the backpressure valve. When the supervisor falls behind, the
queue fills and the *sampler* sheds fps — never the decoder, and never the store. Only
small `RawEvent`s cross the process boundary; a frame never does.

Implements P2.7.
"""

from __future__ import annotations

from muster.config.schema import MusterConfig


class Supervisor:
    """Runs one site: N camera workers plus the shared aggregator, store, and sinks."""

    def __init__(self, config: MusterConfig) -> None:
        raise NotImplementedError

    async def run(self) -> None:
        """Run until shutdown, restarting failed workers with backoff."""
        raise NotImplementedError

    async def reload(self, config: MusterConfig) -> None:
        """Validate-then-swap: rebuild geometry and push new budgets without dropping streams."""
        raise NotImplementedError
