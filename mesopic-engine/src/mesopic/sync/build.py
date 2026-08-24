"""Turn `cloud_sync:` into a running loop, or into a refusal the operator can act on.

Implements part of C5.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from mesopic.errors import ConfigError
from mesopic.sync.client import MetricsSyncClient
from mesopic.sync.loop import MetricsSource, SyncLoop

if TYPE_CHECKING:
    from mesopic.config.schema import CloudSyncConfig


def build_sync_loop(settings: CloudSyncConfig, source: MetricsSource) -> SyncLoop | None:
    """The loop this config asks for, or `None` when cloud sync is off.

    `None` rather than a disabled loop: off is the default and the whole local product
    works without ever reaching the network (ADR-0001), so there should be nothing running
    to inspect, wake, or accidentally re-enable.
    """
    if not settings.enabled:
        return None
    # Both are guaranteed by the config validator, which refuses an enabled sync without
    # an https endpoint and a token reference. Re-checked because this function is also
    # reachable from a hand-built config object in a test or a script.
    if not settings.endpoint or not settings.site_token_env:
        msg = "cloud sync is enabled but has no endpoint or site_token_env"
        raise ConfigError(msg)
    client = MetricsSyncClient(settings.endpoint, _site_token(settings.site_token_env))
    return SyncLoop(source, client)


def _site_token(env_var: str) -> str:
    """Read the per-site bearer token from the environment.

    Refusing at load rather than starting and failing every request: the latter is an
    engine that looks healthy while `unsynced_count` climbs for a week, and the fault is
    in a deploy the operator can fix in a second if anyone tells them.
    """
    token = os.environ.get(env_var)
    if not token:
        # Names the variable, never its value.
        msg = f"${env_var} is unset or empty, and cloud sync needs it to authenticate"
        raise ConfigError(msg)
    return token
