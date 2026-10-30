import sqlite3
from contextlib import closing
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from fastapi_users import InvalidPasswordException
from httpx_oauth.oauth2 import BaseOAuth2, OAuth2Token

from backend.api.accounts import Accounts, GoogleOIDC, owner_of
from backend.database.accounts import scrub_copy

# https, because the client only sends a Secure cookie back over https
PUBLIC_URL = "https://testserver"
SECRET = "s" * 32
EMAIL = "asha@example.com"
PASSWORD = "tea at four on the terrace"
GOOD_ENV = {"STOCKINTEL_AUTH_SECRET": SECRET, "STOCKINTEL_PUBLIC_URL": "https://you.duckdns.org"}


class FakeGoogle(GoogleOIDC):
    """Our Google client with the network parts canned: discovery, code exchange and profile."""

    def __init__(self):
        BaseOAuth2.__init__(self, "client-id", "client-secret", "https://accounts.example/auth",
                            "https://accounts.example/token", name="google")
        self.profile = {"sub": "google-1", "email": "asha@gmail.com", "email_verified": True}

    async def get_access_token(self, code, redirect_uri, code_verifier=None):
        return OAuth2Token({"access_token": "google-access", "refresh_token": "google-refresh", "expires_in": 3600})

    async def get_profile(self, token):
        return self.profile


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "stockintel.db")


@pytest.fixture
def google():
    return FakeGoogle()


@pytest.fixture
def accounts(db_path, google):
    return Accounts(db_path, PUBLIC_URL, SECRET, google)


def client_for(accounts):
    app = FastAPI()
    accounts.include(app, dependencies=[])

    @app.get("/probe")
    def probe(user=Depends(accounts.current_user)):
        return {"owner": owner_of(user)}

    @app.delete("/me")
    async def delete_me(user=Depends(accounts.current_user), manager=Depends(accounts.users.get_user_manager)):
        await manager.delete(user)

    return TestClient(app, base_url=accounts.public_url)


@pytest.fixture
def client(accounts):
    return client_for(accounts)


def sign_in(client, accounts, email=EMAIL):
    accounts.create_user(email, PASSWORD, verified=True)
    return client.post("/auth/cookie/login", data={"username": email, "password": PASSWORD})


def google_sign_in(client):
    url = client.get("/auth/google/authorize").json()["authorization_url"]
    return client.get("/auth/google/callback", params={"code": "code", "state": parse_qs(urlsplit(url).query)["state"][0]})


def rows(db_path, table):
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]


def age_sessions(db_path, days):
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("UPDATE accesstoken SET created_at = datetime(created_at, ?)", (f"-{days} days",))


def test_login_sets_a_host_only_secure_http_only_lax_cookie(client, accounts):
    name, *attributes = [part.strip() for part in sign_in(client, accounts).headers["set-cookie"].split(";")]
    assert name.startswith("__Host-stockintel=")
    assert {"HttpOnly", "Secure", "SameSite=lax", "Path=/"} <= set(attributes)
    assert not [a for a in attributes if a.lower().startswith("domain")]


def test_the_login_cookie_lasts_30_days(client, accounts):
    assert "Max-Age=2592000" in sign_in(client, accounts).headers["set-cookie"]


def test_loopback_http_gets_a_plain_cookie_without_secure(db_path):
    local = Accounts(db_path, "http://127.0.0.1:8931", SECRET, None)
    cookie = sign_in(client_for(local), local).headers["set-cookie"]
    assert cookie.startswith("stockintel=") and "Secure" not in cookie


def test_a_signed_in_cookie_reaches_the_probe_as_its_owner(client, accounts):
    user = accounts.create_user(EMAIL, PASSWORD, verified=True)
    client.post("/auth/cookie/login", data={"username": EMAIL, "password": PASSWORD})
    assert client.get("/probe").json() == {"owner": f"user:{user.id}"}


def test_anonymous_probe_gets_401(client):
    assert client.get("/probe").status_code == 401


def test_a_forged_cookie_gets_401(client, accounts):
    sign_in(client_for(accounts), accounts)
    assert client.get("/probe", headers={"Cookie": "__Host-stockintel=forged"}).status_code == 401


def test_the_old_cookie_gets_401_after_logout(client, accounts):
    sign_in(client, accounts)
    token = client.cookies["__Host-stockintel"]
    assert client.post("/auth/cookie/logout").status_code == 204
    assert client.get("/probe", headers={"Cookie": f"__Host-stockintel={token}"}).status_code == 401


def test_a_session_older_than_session_days_gets_401(client, accounts, db_path):
    sign_in(client, accounts)
    age_sessions(db_path, 31)
    assert client.get("/probe").status_code == 401


def test_expired_sessions_are_purged_at_start(client, accounts, db_path, google):
    sign_in(client, accounts)
    age_sessions(db_path, 31)
    Accounts(db_path, PUBLIC_URL, SECRET, google)
    assert rows(db_path, "accesstoken") == 0


@pytest.mark.parametrize("column, status", [("is_verified", 403), ("is_active", 401)])
def test_a_session_of_a_user_no_longer_verified_or_active_is_refused(client, accounts, db_path, column, status):
    sign_in(client, accounts)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(f'UPDATE "user" SET {column} = 0')
    assert client.get("/probe").status_code == status


def test_an_unverified_user_cannot_log_in(client, accounts):
    accounts.create_user(EMAIL, PASSWORD, verified=False)
    r = client.post("/auth/cookie/login", data={"username": EMAIL, "password": PASSWORD})
    assert (r.status_code, r.json()["detail"]) == (400, "LOGIN_USER_NOT_VERIFIED")


def test_google_sign_in_creates_a_verified_user(client):
    assert google_sign_in(client).status_code == 204
    assert client.get("/probe").status_code == 200


def test_google_sign_in_works_over_loopback_http(db_path, google):
    assert google_sign_in(client_for(Accounts(db_path, "http://127.0.0.1:8931", SECRET, google))).status_code == 204


def test_google_tokens_are_not_stored(client, db_path):
    google_sign_in(client)
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT access_token, refresh_token FROM oauth_account").fetchall() == [("", None)]


@pytest.mark.parametrize("verified", [False, "false"])
def test_an_unverified_google_email_is_refused(client, google, verified):
    google.profile["email_verified"] = verified
    r = google_sign_in(client)
    assert (r.status_code, r.json()["detail"]) == (400, "OAUTH_NOT_AVAILABLE_EMAIL")


def test_google_cannot_take_over_a_password_account_with_the_same_email(client, accounts, google):
    accounts.create_user(google.profile["email"], PASSWORD, verified=True)
    r = google_sign_in(client)
    assert (r.status_code, r.json()["detail"]) == (400, "OAUTH_USER_ALREADY_EXISTS")


def test_the_google_callback_without_the_csrf_cookie_is_refused(client):
    url = client.get("/auth/google/authorize").json()["authorization_url"]
    client.cookies.clear()
    r = client.get("/auth/google/callback", params={"code": "code", "state": parse_qs(urlsplit(url).query)["state"][0]})
    assert (r.status_code, r.json()["detail"]) == (400, "OAUTH_INVALID_STATE")


def test_deleting_a_user_deletes_its_google_link_and_sessions(client, db_path):
    google_sign_in(client)
    client.delete("/me")
    assert [rows(db_path, t) for t in ("user", "oauth_account", "accesstoken")] == [0, 0, 0]


def set_env(monkeypatch, env):
    for name in [*GOOD_ENV, "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"]:
        if env.get(name) is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, env[name])


@pytest.mark.parametrize("change, named", [
    ({"STOCKINTEL_AUTH_SECRET": None}, "STOCKINTEL_AUTH_SECRET"),
    ({"STOCKINTEL_AUTH_SECRET": "s" * 31}, "STOCKINTEL_AUTH_SECRET"),
    ({"STOCKINTEL_PUBLIC_URL": None}, "STOCKINTEL_PUBLIC_URL"),
    ({"STOCKINTEL_PUBLIC_URL": "http://you.duckdns.org"}, "STOCKINTEL_PUBLIC_URL"),
    ({"STOCKINTEL_PUBLIC_URL": "https://you.duckdns.org/"}, "STOCKINTEL_PUBLIC_URL"),
    ({"STOCKINTEL_PUBLIC_URL": "https://a@you.duckdns.org"}, "STOCKINTEL_PUBLIC_URL"),
    ({"STOCKINTEL_PUBLIC_URL": "https://you.duckdns.org:443"}, "STOCKINTEL_PUBLIC_URL"),
    ({"STOCKINTEL_PUBLIC_URL": "https://You.duckdns.org"}, "STOCKINTEL_PUBLIC_URL"),
    ({"GOOGLE_CLIENT_ID": "client-id"}, "GOOGLE_CLIENT_SECRET"),
    ({"GOOGLE_CLIENT_SECRET": "client-secret"}, "GOOGLE_CLIENT_ID"),
])
def test_from_env_names_the_missing_or_bad_variable(monkeypatch, db_path, change, named):
    set_env(monkeypatch, {**GOOD_ENV, **change})
    with pytest.raises(RuntimeError, match=f"^{named}"):
        Accounts.from_env(db_path)


def test_from_env_accepts_http_on_loopback(monkeypatch, db_path):
    set_env(monkeypatch, {**GOOD_ENV, "STOCKINTEL_PUBLIC_URL": "http://127.0.0.1:8931"})
    assert Accounts.from_env(db_path).public_url == "http://127.0.0.1:8931"


def test_without_google_set_up_no_google_route_is_mounted(monkeypatch, db_path):
    set_env(monkeypatch, GOOD_ENV)
    assert client_for(Accounts.from_env(db_path)).get("/auth/google/authorize").status_code == 404


def test_the_stored_hash_uses_the_owasp_argon2_cost(accounts, db_path):
    accounts.create_user(EMAIL, PASSWORD, verified=True)
    with closing(sqlite3.connect(db_path)) as conn:
        assert "$m=19456,t=2,p=1$" in conn.execute('SELECT hashed_password FROM "user"').fetchone()[0]


@pytest.mark.parametrize("password", ["q" * 14, "q" * 65,
                                      "my name is asha, hello", "MY NAME IS ASHA, HELLO"])
def test_a_weak_password_is_refused(accounts, password):
    with pytest.raises(InvalidPasswordException):
        accounts.create_user(EMAIL, password, verified=True)


@pytest.mark.parametrize("length", [15, 64])
def test_passwords_at_the_length_limits_are_accepted(accounts, length):
    assert accounts.create_user(EMAIL, "q" * length, verified=True).email == EMAIL


def test_scrub_copy_removes_sessions_and_keeps_users(client, accounts, db_path):
    sign_in(client, accounts)
    with closing(sqlite3.connect(db_path)) as live, closing(sqlite3.connect(":memory:")) as copy:
        live.backup(copy)
        scrub_copy(copy)
        assert [copy.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in ("accesstoken", "user")] == [0, 1]


def test_scrub_copy_leaves_a_database_from_before_sign_in_alone():
    with closing(sqlite3.connect(":memory:")) as copy:
        copy.execute("CREATE TABLE portfolios (name TEXT)")
        scrub_copy(copy)
        assert copy.execute("SELECT name FROM sqlite_master").fetchall() == [("portfolios",)]
