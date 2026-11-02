"""Sign-in with fastapi-users. The session is a cookie naming a database row, so signing out really ends it."""
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
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.mail import SMTP_SETTINGS, SmtpMailer, mailer_from_env
from backend.database.accounts import AccessToken, OAuthAccount, User, create_tables, purge_sessions, sessions

PASSWORD_MIN = 15           # NIST SP 800-63B-4, for a password that is the only factor
PASSWORD_MAX = 64
SESSION_SECONDS = 30 * 86400  # PROVISIONAL: how long a sign-in lasts; shorten if people share devices
SECRET_MIN = 32
LOOPBACK = ("127.0.0.1", "localhost")
GOOGLE_DISCOVERY = "https://accounts.google.com/.well-known/openid-configuration"
# OWASP's Argon2id minimum. Measured on an M4 Mac: a hash adds 20 MB to peak memory and uses 15 ms of CPU,
# against 67 MB and 95 ms at the library default. fastapi-users hashes on the event loop, one at a time.
HASHER = Argon2Hasher(memory_cost=19456, time_cost=2, parallelism=1)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    def __init__(self, user_db: SQLAlchemyUserDatabase, passwords: PasswordHelper, secret: str, public_url: str,
                 mailer: Optional[SmtpMailer]):
        super().__init__(user_db, passwords)
        self.verification_token_secret = self.reset_password_token_secret = secret
        self.public_url, self.mailer = public_url, mailer

    async def on_after_register(self, user: User, request: Optional[Request] = None) -> None:
        if not user.is_verified:
            await self.request_verify(user, request)

    # Links use public_url, not the request's Host, and keep the token after '#', out of logs and Referers
    async def on_after_request_verify(self, user: User, token: str, request: Optional[Request] = None) -> None:
        await self.mailer.send(user.email, "Confirm your email for StockIntel", (
            f"Open this link to confirm your email:\n{self.public_url}/#/verify?token={token}\n\n"
            f"It works for {self.verification_token_lifetime_seconds // 60} minutes. "
            "If you did not sign up for StockIntel, you can ignore this email."))

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[Request] = None) -> None:
        await self.mailer.send(user.email, "Reset your StockIntel password", (
            f"Open this link to choose a new password:\n{self.public_url}/#/reset?token={token}\n\n"
            f"It works for {self.reset_password_token_lifetime_seconds // 60} minutes. A new password signs you "
            "out on every device. If you did not ask for this, you can ignore this email."))

    async def on_after_reset_password(self, user: User, request: Optional[Request] = None) -> None:
        await self.user_db.session.execute(delete(AccessToken).where(AccessToken.user_id == user.id))
        await self.user_db.session.commit()

    async def validate_password(self, password: str, user: Union[schemas.BaseUserCreate, User]) -> None:
        if not PASSWORD_MIN <= len(password) <= PASSWORD_MAX:
            raise InvalidPasswordException(f"Use {PASSWORD_MIN} to {PASSWORD_MAX} characters (this one has "
                                           f"{len(password)}). A short sentence is easy to remember.")
        name = user.email.split("@")[0].lower()
        if len(name) > 1 and name in password.lower():
            raise InvalidPasswordException(f"Your password must not contain {name!r}, the start of your email.")

    async def oauth_callback(self, oauth_name: str, access_token: str, account_id: str, account_email: str,
                             expires_at: Optional[int] = None, refresh_token: Optional[str] = None,
                             request: Optional[Request] = None, **flags: bool) -> User:
        holder = await self.user_db.get_by_email(account_email)
        # Whoever registered this address by password never proved it is theirs, so they must not block its owner
        if holder is not None and not holder.is_verified and not holder.oauth_accounts:
            await self.user_db.delete(holder)
        # We never call Google after sign-in, so its tokens are neither kept nor backed up
        return await super().oauth_callback(oauth_name, "", account_id, account_email, expires_at, None,
                                            request, **flags)


class GoogleOIDC(OpenID):
    def __init__(self, client_id: str, client_secret: str):
        super().__init__(client_id, client_secret, GOOGLE_DISCOVERY, name="google", base_scopes=["openid", "email"])

    async def get_id_email(self, token: str) -> Tuple[str, Optional[str]]:
        profile = await self.get_profile(token)
        # no email makes fastapi-users answer 400: an address Google has not verified proves nothing
        return str(profile["sub"]), profile["email"] if profile.get("email_verified") is True else None


def owner_of(user: User) -> str:
    return f"user:{user.id}"


class Accounts:
    def __init__(self, db_path: str, public_url: str, secret: str, google: Optional[BaseOAuth2],
                 mailer: Optional[SmtpMailer]):
        self.public_url, self.secret, self.google, self.mailer = public_url, secret, google, mailer
        self.secure = urlsplit(public_url).scheme == "https"
        create_tables(db_path)
        purge_sessions(db_path, datetime.now(timezone.utc) - timedelta(seconds=SESSION_SECONDS))
        self._sessions = sessions(db_path)
        self._passwords = PasswordHelper(PasswordHash((HASHER,)))

        async def get_session():
            async with self._sessions() as session:
                yield session

        async def get_user_manager(session: AsyncSession = Depends(get_session)):
            yield self._manager(session)

        def get_strategy(session: AsyncSession = Depends(get_session)) -> DatabaseStrategy:
            return DatabaseStrategy(SQLAlchemyAccessTokenDatabase(session, AccessToken),
                                    lifetime_seconds=SESSION_SECONDS)

        # __Host- makes the browser refuse the cookie unless it is Secure, host-only and on path /
        transport = CookieTransport(cookie_name="__Host-stockintel" if self.secure else "stockintel",
                                    cookie_max_age=SESSION_SECONDS, cookie_path="/", cookie_secure=self.secure,
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
        port = "" if parts.port in (None, {"https": 443, "http": 80}.get(parts.scheme)) else f":{parts.port}"
        if public_url != f"{parts.scheme}://{parts.hostname}{port}" or not parts.hostname or not (
                parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK)):
            raise RuntimeError(f"STOCKINTEL_PUBLIC_URL is {public_url!r}; set it to the address people open, as a "
                               "browser's Origin header writes it (lower case; no user name, path, trailing slash "
                               "or default port), like https://you.duckdns.org (http only for 127.0.0.1 "
                               "or localhost).")
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        if bool(client_id) != bool(client_secret):
            missing = "GOOGLE_CLIENT_SECRET" if client_id else "GOOGLE_CLIENT_ID"
            raise RuntimeError(f"{missing} is not set but its pair is; set both from the Google Cloud console's "
                               "OAuth client, or neither to turn Google sign-in off.")
        return cls(db_path, public_url, secret, GoogleOIDC(client_id, client_secret) if client_id else None,
                   mailer_from_env())

    def _manager(self, session: AsyncSession) -> UserManager:
        return UserManager(SQLAlchemyUserDatabase(session, User, OAuthAccount), self._passwords, self.secret,
                           self.public_url, self.mailer)

    def include(self, app: FastAPI, dependencies: Sequence[params.Depends]) -> None:
        """The users router stays out: its PATCH /me changes email and password."""
        app.include_router(self.users.get_auth_router(self.backend, requires_verification=True),
                           prefix="/auth/cookie", dependencies=dependencies)
        if self.google is not None:
            # associate_by_email stays off: a password account made first under someone's Gmail
            # would otherwise take over their later Google sign-in
            app.include_router(self.users.get_oauth_router(
                self.google, self.backend, self.secret, redirect_url=self.public_url + "/login/google",
                associate_by_email=False, is_verified_by_default=True, csrf_token_cookie_secure=self.secure),
                prefix="/auth/google", dependencies=dependencies)
        if self.mailer is not None:
            for router in (self.users.get_register_router(schemas.BaseUser[uuid.UUID], schemas.BaseUserCreate),
                           self.users.get_verify_router(schemas.BaseUser[uuid.UUID]),
                           self.users.get_reset_password_router()):
                app.include_router(router, prefix="/auth", dependencies=dependencies)

    def create_user(self, email: str, password: str, *, verified: bool) -> User:
        if not verified and self.mailer is None:
            raise RuntimeError(f"{email} could never confirm its address: no email sender is set up to send the "
                               f"link. Create the user verified, or set {', '.join(SMTP_SETTINGS)}.")
        async def create() -> User:
            async with self._sessions() as session:
                return await self._manager(session).create(
                    schemas.BaseUserCreate(email=email, password=password, is_verified=verified))
        return asyncio.run(create())

    def user(self, email: str) -> Optional[User]:
        async def find() -> Optional[User]:
            async with self._sessions() as session:
                return await SQLAlchemyUserDatabase(session, User, OAuthAccount).get_by_email(email)
        return asyncio.run(find())

    def has_users(self) -> bool:
        async def any_user() -> bool:
            async with self._sessions() as session:
                return await session.scalar(select(User.id).limit(1)) is not None
        return asyncio.run(any_user())
