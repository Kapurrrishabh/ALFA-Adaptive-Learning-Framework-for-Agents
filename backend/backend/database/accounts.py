"""Sign-in tables for fastapi-users (user, oauth_account, accesstoken), kept in the same
SQLite file as the portfolios so one backup holds both."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List

from fastapi_users_db_sqlalchemy import SQLAlchemyBaseOAuthAccountTableUUID, SQLAlchemyBaseUserTableUUID
from fastapi_users_db_sqlalchemy.access_token import SQLAlchemyBaseAccessTokenTableUUID
from sqlalchemy import URL, create_engine, delete, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, relationship
from sqlalchemy.pool import NullPool


class Base(DeclarativeBase):
    pass


class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    # passive_deletes: let the database cascade remove a deleted user's Google link; without it
    # SQLAlchemy sets the link's user_id to NULL and the delete fails
    oauth_accounts: Mapped[List[OAuthAccount]] = relationship(lazy="joined", passive_deletes="all")


class AccessToken(SQLAlchemyBaseAccessTokenTableUUID, Base):
    pass


def _url(driver: str, path: str) -> URL:
    if path == ":memory:":
        raise ValueError("accounts need a database file: with ':memory:' every connection opens its own empty database")
    return URL.create(driver, database=path)


def _on_connect(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")      # SQLite ignores ON DELETE CASCADE without it
    cursor.execute("PRAGMA busy_timeout = 5000")    # PROVISIONAL: Python's default, pinned; the Store writes to this file too
    cursor.close()


def create_tables(path: str) -> None:
    Base.metadata.create_all(create_engine(_url("sqlite", path), poolclass=NullPool))


def purge_sessions(path: str, older_than: datetime) -> None:
    with create_engine(_url("sqlite", path), poolclass=NullPool).begin() as conn:
        conn.execute(delete(AccessToken).where(AccessToken.created_at < older_than))


def sessions(path: str) -> async_sessionmaker[AsyncSession]:
    # No pool: the CLI and the test client each run their own event loop, and a pooled
    # aiosqlite connection belongs to the loop that opened it.
    engine = create_async_engine(_url("sqlite+aiosqlite", path), poolclass=NullPool)
    event.listen(engine.sync_engine, "connect", _on_connect)
    return async_sessionmaker(engine, expire_on_commit=False)


def scrub_copy(conn: sqlite3.Connection) -> None:
    """Delete the sign-in sessions from a copy of the database, so a backup never holds one."""
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (AccessToken.__tablename__,)).fetchone():
        with conn:
            conn.execute(f"DELETE FROM {AccessToken.__tablename__}")
