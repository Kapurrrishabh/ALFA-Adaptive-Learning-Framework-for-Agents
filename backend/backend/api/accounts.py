"""Sign-in with fastapi-users: Google (OpenID Connect) and email + password. The session is
a cookie naming a database row, so signing out really ends it. Built here, mounted by app.py."""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Tuple, Union
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Request, params
from fastapi_users import BaseUserManager, FastAPIUsers, InvalidPasswordException, UUIDIDMixin, schemas
from fastapi_users.authentication import AuthenticationBackend, CookieTransport
from fastapi_users.authentication.strategy.db import DatabaseStrategy
from fastapi_users.password import PasswordHelper
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyAccessTokenDatabase
from httpx_oauth.clients.openid import OpenID
from httpx_oauth.oauth2 import BaseOAuth2
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.accounts import AccessToken, OAuthAccount, User, create_tables, purge_sessions, sessions

PASSWORD_MIN = 15           # NIST SP 800-63B-4, for a password that is the only factor
PASSWORD_MAX = 64           # NIST asks to allow at least 64
SESSION_DAYS = 30           # PROVISIONAL: how long a sign-in lasts; shorten if people share devices
SECRET_MIN = 32
LOOPBACK = ("127.0.0.1", "localhost")
GOOGLE_DISCOVERY = "https://accounts.google.com/.well-known/openid-configuration"
# The OWASP minimum for Argon2id: about 20 MB and 22 ms a hash (measured on a Mac), against
# 64 MB and 63 ms at the library default. fastapi-users hashes on the event loop, one at a time.
HASHER = Argon2Hasher(memory_cost=19456, time_cost=2, parallelism=1)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    async def validate_password(self, password: str, user: Union[schemas.BaseUserCreate, User]) -> None:
        if not PASSWORD_MIN <= len(password) <= PASSWORD_MAX:
            raise InvalidPasswordException(f"Use {PASSWORD_MIN} to {PASSWORD_MAX} characters (this one has "
                                           f"{len(password)}). A short sentence is easy to remember.")
        name = user.email.split("@")[0].lower()
        if len(name) > 1 and name in password.lower():      # one letter would just ban that letter
            raise InvalidPasswordException(f"Your password must not contain {name!r}, the start of your email.")

    async def oauth_callback(self, oauth_name: str, access_token: str, account_id: str, account_email: str,
                             expires_at: Optional[int] = None, refresh_token: Optional[str] = None,
                             request: Optional[Request] = None, **flags: bool) -> User:
        # We never call Google after sign-in, so its tokens are neither kept nor backed up
        return await super().oauth_callback(oauth_name, "", account_id, account_email, expires_at, None,
                                            request, **flags)


class GoogleOIDC(OpenID):
    """Google with only the openid and email scopes. Building one fetches Google's discovery
    document, so a server with Google set up fails at start when Google is unreachable."""

    def __init__(self, client_id: str, client_secret: str):
        super().__init__(client_id, client_secret, GOOGLE_DISCOVERY, name="google", base_scopes=["openid", "email"])

    async def get_id_email(self, token: str) -> Tuple[str, Optional[str]]:
        profile = await self.get_profile(token)
        # no email makes fastapi-users answer 400: an address Google has not verified proves nothing
        return str(profile["sub"]), profile["email"] if profile.get("email_verified") is True else None


def owner_of(user: User) -> str:
    """The one place a user maps to the portfolio name their data is kept under."""
    return f"user:{user.id}"


class Accounts:
    def __init__(self, db_path: str, public_url: str, secret: str, google: Optional[BaseOAuth2]):
        self.public_url, self.secret, self.google = public_url, secret, google
        self.secure = urlsplit(public_url).scheme == "https"
        create_tables(db_path)
        purge_sessions(db_path, datetime.now(timezone.utc) - timedelta(days=SESSION_DAYS))
        self._sessions = sessions(db_path)
        self._passwords = PasswordHelper(PasswordHash((HASHER,)))

        async def get_session():
            async with self._sessions() as session:
                yield session

        async def get_user_manager(session: AsyncSession = Depends(get_session)):
            yield self._manager(session)

        def get_strategy(session: AsyncSession = Depends(get_session)) -> DatabaseStrategy:
            return DatabaseStrategy(SQLAlchemyAccessTokenDatabase(session, AccessToken),
                                    lifetime_seconds=SESSION_DAYS * 86400)

        # __Host- makes the browser refuse the cookie unless it is Secure, host-only and on path /
        transport = CookieTransport(cookie_name="__Host-stockintel" if self.secure else "stockintel",
                                    cookie_max_age=SESSION_DAYS * 86400, cookie_path="/", cookie_secure=self.secure,
                                    cookie_httponly=True, cookie_samesite="lax")
        self.backend = AuthenticationBackend(name="cookie", transport=transport, get_strategy=get_strategy)
        self.users = FastAPIUsers[User, uuid.UUID](get_user_manager, [self.backend])
        self.current_user = self.users.current_user(active=True, verified=True)

    @classmethod
    def from_env(cls, db_path: str) -> "Accounts":
        secret = os.environ.get("STOCKINTEL_AUTH_SECRET", "")
        if len(secret) < SECRET_MIN:
            raise RuntimeError(f"STOCKINTEL_AUTH_SECRET has {len(secret)} characters and needs at least "
                               f"{SECRET_MIN}; make one with `openssl rand -hex 32`.")
        public_url = os.environ.get("STOCKINTEL_PUBLIC_URL", "")
        parts = urlsplit(public_url)
        if public_url != f"{parts.scheme}://{parts.netloc}" or not parts.hostname or not (
                parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK)):
            raise RuntimeError(f"STOCKINTEL_PUBLIC_URL is {public_url!r}; set it to the address people open, with "
                               "no path or trailing slash, like https://you.duckdns.org (http only for 127.0.0.1 "
                               "or localhost).")
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        if bool(client_id) != bool(client_secret):
            missing = "GOOGLE_CLIENT_SECRET" if client_id else "GOOGLE_CLIENT_ID"
            raise RuntimeError(f"{missing} is not set but its pair is; set both from the Google Cloud console's "
                               "OAuth client, or neither to turn Google sign-in off.")
        return cls(db_path, public_url, secret, GoogleOIDC(client_id, client_secret) if client_id else None)

    def _manager(self, session: AsyncSession) -> UserManager:
        return UserManager(SQLAlchemyUserDatabase(session, User, OAuthAccount), self._passwords)

    def include(self, app: FastAPI, dependencies: Sequence[params.Depends]) -> None:
        """Password sign-in at /auth/cookie, and Google at /auth/google when it is set up. The
        users router stays out: its PATCH /me changes email and password."""
        app.include_router(self.users.get_auth_router(self.backend, requires_verification=True),
                           prefix="/auth/cookie", dependencies=dependencies)
        if self.google is not None:
            # associate_by_email stays off: a password account made first under someone's Gmail
            # would otherwise take over their later Google sign-in
            app.include_router(self.users.get_oauth_router(
                self.google, self.backend, self.secret, redirect_url=self.public_url + "/login/google",
                associate_by_email=False, is_verified_by_default=True, csrf_token_cookie_secure=self.secure),
                prefix="/auth/google", dependencies=dependencies)

    def create_user(self, email: str, password: str, *, verified: bool) -> User:
        """For the command line and tests; it runs its own event loop."""
        async def create() -> User:
            async with self._sessions() as session:
                return await self._manager(session).create(
                    schemas.BaseUserCreate(email=email, password=password, is_verified=verified))
        return asyncio.run(create())
