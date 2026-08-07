"""Emit metrics in the shapes the self-hosted ecosystem already speaks (ADR-0006).

Exporters read from the store or subscribe to freshly committed buckets. They never
compute and never persist.

They are independently enable/disable-able and **fail independently**: a dead MQTT broker
must not stall the Prometheus scrape or the cloud sync. A broken exporter is a degraded
state on `/healthz`, never a pipeline stall.
"""

from __future__ import annotations

from muster.exporters.exporter import Exporter

__all__ = ["Exporter"]
