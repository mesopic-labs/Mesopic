"""Tests for the Runtime vocabulary."""

from muster.types import Runtime


def test_runtime_values_are_stable_wire_strings() -> None:
    """Assert Runtime enum values match the expected wire strings."""
    assert Runtime.ORT_CPU == "ort-cpu"
    assert Runtime.OPENVINO == "openvino"
