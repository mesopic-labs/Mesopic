"""The engine's local credential, and the pieces the login routes are built from.

The engine authenticates *writes* and nothing else (ADR-0019): the counts are not the
sensitive thing on a LAN, the ability to rewrite the site's configuration is. These tests
cover the three primitives — the credential, the session table, the login throttle — and
`test_api_auth_routes.py` covers what they add up to over HTTP.

Every clock here is injected, for the reason the supervisor's clocks are: a test that
waits twelve real hours for a session to expire is a test nobody runs.

Red-first for P3.10.
"""

from __future__ import annotations

import pytest

from muster.api.auth import Credential, LoginThrottle, SessionStore

PASSWORD = "correct-horse-battery-staple"  # noqa: S105 - a fixture, not a credential


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --- The credential -----------------------------------------------------------


def test_the_exact_password_verifies() -> None:
    assert Credential(PASSWORD).verify(PASSWORD)


@pytest.mark.parametrize(
    "submitted",
    [
        "",
        "wrong",
        "correct-horse-battery-stapl",
        "correct-horse-battery-staple ",
        "CORRECT-HORSE-BATTERY-STAPLE",
    ],
    ids=["empty", "wrong", "prefix", "trailing-space", "case"],
)
def test_anything_else_is_refused(submitted: str) -> None:
    assert not Credential(PASSWORD).verify(submitted)


def test_a_password_outside_ascii_verifies() -> None:
    """The comparison is over encoded bytes, so a non-ASCII password must round-trip.

    `hmac.compare_digest` refuses `str` arguments that are not ASCII-only, so a naive
    implementation raises `TypeError` here rather than returning a bool — which a route
    would surface as a 500 on a correct password.
    """
    assert Credential("pässwörd-Ω").verify("pässwörd-Ω")


# --- The session table --------------------------------------------------------


def test_an_issued_session_is_valid() -> None:
    sessions = SessionStore()

    token = sessions.issue()

    assert sessions.is_valid(token)


def test_every_session_token_is_distinct() -> None:
    sessions = SessionStore()

    issued = {sessions.issue() for _ in range(16)}

    assert len(issued) == 16


def test_a_token_nobody_issued_is_not_valid() -> None:
    sessions = SessionStore()
    sessions.issue()

    assert not sessions.is_valid("a-token-from-somewhere-else")


def test_a_missing_cookie_is_not_valid() -> None:
    """The route hands this whatever the cookie jar held, which is `None` when unset."""
    assert not SessionStore().is_valid(None)


def test_a_session_expires_on_an_absolute_deadline() -> None:
    clock = FakeClock()
    sessions = SessionStore(monotonic=clock, ttl_s=100.0)
    token = sessions.issue()

    clock.advance(101.0)

    assert not sessions.is_valid(token)


def test_using_a_session_does_not_extend_it() -> None:
    """No sliding renewal: 12 hours means 12 hours, not 12 hours after the last click."""
    clock = FakeClock()
    sessions = SessionStore(monotonic=clock, ttl_s=100.0)
    token = sessions.issue()

    clock.advance(60.0)
    assert sessions.is_valid(token)
    clock.advance(60.0)

    assert not sessions.is_valid(token)


def test_a_revoked_session_stops_working() -> None:
    sessions = SessionStore()
    token = sessions.issue()

    sessions.revoke(token)

    assert not sessions.is_valid(token)


def test_revoking_a_token_nobody_issued_is_not_an_error() -> None:
    """Logout runs on whatever cookie arrived, including a stale or invented one."""
    SessionStore().revoke("not-a-session")


def test_the_session_table_is_bounded() -> None:
    """An operator who logs in repeatedly must not grow the table without limit.

    Nobody can issue a session without the password, so this is not an attack surface —
    it is the ordinary "left a tab open for a month" case, and an unbounded dict on a box
    with 8 GB of RAM is still a leak.
    """
    sessions = SessionStore(capacity=4)

    tokens = [sessions.issue() for _ in range(6)]

    assert not sessions.is_valid(tokens[0])
    assert sessions.is_valid(tokens[-1])
    assert sum(sessions.is_valid(token) for token in tokens) == 4


# --- The login throttle -------------------------------------------------------


def test_a_fresh_throttle_permits_a_login() -> None:
    assert not LoginThrottle().blocked()


def test_failures_below_the_limit_do_not_block() -> None:
    throttle = LoginThrottle(max_failures=3)

    for _ in range(2):
        throttle.record_failure()

    assert not throttle.blocked()


def test_the_limit_blocks_further_attempts() -> None:
    throttle = LoginThrottle(max_failures=3)

    for _ in range(3):
        throttle.record_failure()

    assert throttle.blocked()


def test_the_block_lifts_after_the_cooldown() -> None:
    clock = FakeClock()
    throttle = LoginThrottle(monotonic=clock, max_failures=1, cooldown_s=60.0)
    throttle.record_failure()

    clock.advance(61.0)

    assert not throttle.blocked()


def test_a_success_clears_the_failures() -> None:
    """Otherwise a week of typos eventually locks out an operator who never got it wrong twice."""
    throttle = LoginThrottle(max_failures=3)
    throttle.record_failure()
    throttle.record_failure()

    throttle.succeeded()

    assert not throttle.blocked()
    throttle.record_failure()
    throttle.record_failure()
    assert not throttle.blocked()
