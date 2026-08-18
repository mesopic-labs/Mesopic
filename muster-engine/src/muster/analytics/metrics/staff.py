"""The one rule every metric's staff sub-count follows (adjacency A2).

**`value` is the total, staff included. `staff_value` is the staff portion. Customers are
the difference.** Decided in ADR-0021, and it is the reason this convention is written
once here rather than four times in four reducers: the alternative — making `value` mean
customers — would silently change what every row already in `metrics_minute` means, and
what the ADR-0010 sync contract has been shipping.

Two shapes, because the events have two shapes:

* A **transition** is a fact about one track and carries `is_staff`, so its staff
  sub-count is the same reduction run over the subset — `staff_only` below.
* A **sampled state** counts a zone and names no track, so there is nothing to filter on
  and the split rides the sample itself (`RawEvent.staff_value`, ADR-0016's precedent).

One distinction worth keeping straight when a reducer implements this: a **count** of no
staff is `0.0`, because a count of nobody is zero; a **mean** over no staff samples is
`None`, because the average of nothing is not zero, and a dwell row claiming staff stayed
for zero seconds is a different and wronger statement than one admitting no staff stayed.

Implements P4.2.
"""

from __future__ import annotations

from collections.abc import Sequence

from muster.types import RawEvent


def staff_only(events: Sequence[RawEvent]) -> list[RawEvent]:
    """The subset a transition metric reduces to get its staff sub-count."""
    return [event for event in events if event.is_staff]
