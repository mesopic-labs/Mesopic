"""The engine's local credential: one password, one session table, one throttle.

ADR-0019 settles the posture this implements. The short version, because it is easy to
read this module as smaller than it should be:

* **Writes are authenticated, reads are not.** A LAN dashboard that needs a password to
  look at is a worse product, and the counts are not the sensitive thing here — the
  ability to rewrite `muster.yaml` is (§13, ADR-0018 item 8).
* **The password is an env reference in config, never a value in it** — the rule every
  other secret in `muster.yaml` follows.
* **No credential configured means writes refuse.** There is deliberately no "open while
  bound to loopback" path: that makes the posture depend on a value that does not measure
  real exposure, and P3.6's container binds every interface regardless.

The session cookie is opaque and the table behind it lives in memory, so a restart
invalidates every session and the engine gains no new persistent state — in particular
nothing the sync client could ever see.

Implements P3.10.
"""

from __future__ import annotations

import hmac
import logging
import secrets
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from muster.api.forms import bounded_body, declared_too_large, form_field

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

SESSION_COOKIE = "muster_session"
"""Named rather than defaulted, because logout has to clear exactly what login set."""

SESSION_TTL_S = 12 * 60 * 60
"""How long a login lasts, absolutely — there is no sliding renewal.

A shift is the unit an operator thinks in, and a session that silently renews forever is
one an unattended browser keeps alive indefinitely."""

MAX_SESSIONS = 32
"""Sessions held at once. Nobody can issue one without the password, so this is not an
attack surface; it bounds the ordinary case of an operator who logs in from a new tab
every day and never logs out."""

MAX_LOGIN_FAILURES = 5
LOGIN_COOLDOWN_S = 60.0
"""Failures before the endpoint refuses everything, and for how long.

Global rather than per-IP: keying by address means an unbounded dict fed by untrusted
input, for a threat model where the attacker is already on the LAN. The cost is stated in
ADR-0019 item 7 rather than hidden — someone on the LAN can lock the operator out for a
minute."""

TOKEN_BYTES = 32

MAX_LOGIN_BODY_BYTES = 4096
"""Ceiling on a login body, checked before anything parses it.

A password field is the one place on this surface where a stranger chooses how many bytes
the engine allocates, and "explicit limits on everything unbounded" applies to a form as
much as to a batch or a polygon."""

GENERIC_LOGIN_REFUSAL = "that did not work"
"""What every failed login says, whatever went wrong.

Distinguishing "wrong password" from "too many attempts" tells a caller that the password
they just tried was the right one — which is the single fact the throttle exists to
withhold."""


class Credential:
    """The operator's password, and the only thing that may compare against it."""

    def __init__(self, password: str) -> None:
        self._expected = password.encode("utf-8")

    def verify(self, submitted: str) -> bool:
        """Constant-time comparison of what was typed against what was configured.

        The encode is not incidental: `hmac.compare_digest` rejects `str` arguments that
        are not ASCII-only, so comparing text would raise on a perfectly good non-ASCII
        password — a 500 on the correct credential.
        """
        return hmac.compare_digest(self._expected, submitted.encode("utf-8"))


class SessionStore:
    """Opaque session tokens and their absolute deadlines.

    The token carries nothing — it is an index into this table — so there is no claim to
    forge and no signing key to manage or rotate.
    """

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        ttl_s: float = SESSION_TTL_S,
        capacity: int = MAX_SESSIONS,
    ) -> None:
        self._monotonic = monotonic
        self._ttl_s = ttl_s
        self._capacity = capacity
        self._expiries: dict[str, float] = {}

    def issue(self) -> str:
        """Mint a session. Only a verified password may reach this."""
        self._drop_expired()
        while len(self._expiries) >= self._capacity:
            self._expiries.pop(next(iter(self._expiries)))
        token = secrets.token_urlsafe(TOKEN_BYTES)
        self._expiries[token] = self._monotonic() + self._ttl_s
        return token

    def is_valid(self, token: str | None) -> bool:
        """`None` is what an unset cookie looks like by the time it reaches here."""
        if token is None:
            return False
        expiry = self._expiries.get(token)
        if expiry is None:
            return False
        if self._monotonic() >= expiry:
            del self._expiries[token]
            return False
        return True

    def revoke(self, token: str | None) -> None:
        """Forget a session. Logout runs on whatever cookie arrived, so this forgives."""
        if token is not None:
            self._expiries.pop(token, None)

    def _drop_expired(self) -> None:
        now = self._monotonic()
        for token in [token for token, expiry in self._expiries.items() if now >= expiry]:
            del self._expiries[token]


class LoginThrottle:
    """A bounded number of failures, then a cooldown during which nothing is accepted.

    Refusing the *correct* password during the cooldown is the point: a throttle that
    lets a right answer through is a throttle that can be brute-forced at the rate the
    attacker guesses right.
    """

    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        max_failures: int = MAX_LOGIN_FAILURES,
        cooldown_s: float = LOGIN_COOLDOWN_S,
    ) -> None:
        self._monotonic = monotonic
        self._max_failures = max_failures
        self._cooldown_s = cooldown_s
        self._failures = 0
        self._blocked_until = 0.0

    def blocked(self) -> bool:
        if self._monotonic() < self._blocked_until:
            return True
        if self._blocked_until:
            self.succeeded()
        return False

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._max_failures:
            self._blocked_until = self._monotonic() + self._cooldown_s

    def succeeded(self) -> None:
        """Clear the count, so a month of occasional typos never adds up to a lockout."""
        self._failures = 0
        self._blocked_until = 0.0


class WriteGuard:
    """What every state-changing route depends on, and the only thing that admits one.

    A dependency rather than middleware, so a route declares its own protection the way it
    declares everything else — and `test_every_write_route_requires_a_session` walks the
    route table so a future write route that forgets to declare it fails at merge time.
    """

    def __init__(
        self,
        *,
        credential: Credential | None,
        sessions: SessionStore,
        throttle: LoginThrottle,
    ) -> None:
        self.credential = credential
        self.sessions = sessions
        self.throttle = throttle

    @property
    def configured(self) -> bool:
        """Whether a credential exists at all. Drives what the calibration page offers."""
        return self.credential is not None

    async def __call__(self, request: Request) -> None:
        """Refuse anything without a live session. Raises, or returns nothing.

        The two refusals are deliberately different codes. 503 is the honest answer when
        no credential is configured: the surface is real, no session could ever satisfy
        it, and answering 401 would send the operator to a login page that cannot work.
        """
        if self.credential is None:
            raise HTTPException(status_code=503, detail="no credential is configured")
        if not self.sessions.is_valid(request.cookies.get(SESSION_COOKIE)):
            raise HTTPException(status_code=401, detail="authentication required")


def auth_router(
    *, guard: WriteGuard, templates: Jinja2Templates, site_id: Callable[[], str]
) -> APIRouter:
    """`GET /login`, `POST /login`, `POST /logout` — registered only when a credential exists.

    An engine with no `api.password_env` serves no login form at all: a form that cannot
    succeed is a worse answer than a 404.
    """
    router = APIRouter()

    def _page(request: Request, *, error: str | None = None, status_code: int = 200) -> Any:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "site_id": site_id(),
                "error": error,
                "authenticated": guard.sessions.is_valid(request.cookies.get(SESSION_COOKIE)),
            },
            status_code=status_code,
        )

    @router.get("/login")
    async def login_page(request: Request) -> Any:
        return _page(request)

    @router.post("/login")
    async def login(request: Request) -> Any:
        """Verify a password and open a session, or refuse without saying why."""
        if declared_too_large(request, MAX_LOGIN_BODY_BYTES):
            raise HTTPException(status_code=413, detail="that request was too large")
        if guard.throttle.blocked():
            # Not recorded as a failure: counting attempts made *during* a cooldown would
            # push its end further out on every try, turning a throttle into a lockout
            # that a flood could hold open indefinitely.
            logger.warning("refused a login attempt during the cooldown")
            return _page(request, error=GENERIC_LOGIN_REFUSAL, status_code=401)

        body = await bounded_body(request, MAX_LOGIN_BODY_BYTES)
        if body is None:
            raise HTTPException(status_code=413, detail="that request was too large")

        submitted = form_field(request, body, "password")
        if submitted is None or guard.credential is None:
            guard.throttle.record_failure()
            logger.info("refused a login carrying no password field")
            return _page(request, error=GENERIC_LOGIN_REFUSAL, status_code=401)
        if not guard.credential.verify(submitted):
            guard.throttle.record_failure()
            # Names neither the value nor its length: both narrow a guess.
            logger.warning("refused a login with an incorrect password")
            return _page(request, error=GENERIC_LOGIN_REFUSAL, status_code=401)

        guard.throttle.succeeded()
        return _session_response("/", token=guard.sessions.issue(), request=request)

    @router.post("/logout")
    async def logout(request: Request) -> Response:
        """Forgiving by design: logout runs on whatever cookie arrived, valid or not."""
        guard.sessions.revoke(request.cookies.get(SESSION_COOKIE))
        response = RedirectResponse(url="/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE, path="/")
        return response

    return router


def _session_response(url: str, *, token: str, request: Request) -> Response:
    response = RedirectResponse(url=url, status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(SESSION_TTL_S),
        path="/",
        httponly=True,
        samesite="strict",
        # Conditional, not unconditional: the LAN default is plain http, and a `Secure`
        # cookie there is set once and never sent back — a login loop with no error.
        secure=request.url.scheme == "https",
    )
    return response


__all__ = [
    "GENERIC_LOGIN_REFUSAL",
    "LOGIN_COOLDOWN_S",
    "MAX_LOGIN_BODY_BYTES",
    "MAX_LOGIN_FAILURES",
    "MAX_SESSIONS",
    "SESSION_COOKIE",
    "SESSION_TTL_S",
    "Credential",
    "LoginThrottle",
    "SessionStore",
    "WriteGuard",
    "auth_router",
]
