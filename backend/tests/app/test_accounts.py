import asyncio
import smtplib
import sqlite3
import ssl
from contextlib import closing
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from fastapi_users import InvalidPasswordException

from backend.api.accounts import Accounts, owner_of
from backend.api.mail import SMTP_TIMEOUT_S, mailer_from_env
from backend.database.accounts import scrub_copy

from .conftest import EMAIL, PASSWORD, PUBLIC_URL, SECRET, sign_in

GOOD_ENV = {"STOCKINTEL_AUTH_SECRET": SECRET, "STOCKINTEL_PUBLIC_URL": "https://you.duckdns.org"}
SMTP_ENV = {"STOCKINTEL_SMTP_HOST": "smtp.example.com", "STOCKINTEL_SMTP_PORT": "465",
            "STOCKINTEL_SMTP_USER": "apikey", "STOCKINTEL_SMTP_PASSWORD": "smtp-password",
            "STOCKINTEL_SMTP_FROM": "StockIntel <noreply@example.com>"}


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


def google_sign_in(client):
    url = client.get("/auth/google/authorize").json()["authorization_url"]
    return client.get("/auth/google/callback", params={"code": "code", "state": parse_qs(urlsplit(url).query)["state"][0]})


def link_sent(outbox):
    return next(word for word in outbox.sent[-1].get_content().split() if "token=" in word)


def token_sent(outbox):
    return link_sent(outbox).split("token=", 1)[1]


def register(client, **body):
    return client.post("/auth/register", json={"email": EMAIL, "password": PASSWORD, **body})


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
    local = Accounts(db_path, "http://127.0.0.1:8931", SECRET, None, None)
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
    Accounts(db_path, PUBLIC_URL, SECRET, google, None)
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


def test_a_sign_up_that_follows_the_emailed_link_can_log_in(client, outbox):
    register(client)
    assert client.post("/auth/verify", json={"token": token_sent(outbox)}).status_code == 200
    client.post("/auth/cookie/login", data={"username": EMAIL, "password": PASSWORD})
    assert client.get("/probe").status_code == 200


def test_a_sign_up_cannot_log_in_before_confirming_even_if_it_claims_to_be_verified(client):
    register(client, is_verified=True)
    r = client.post("/auth/cookie/login", data={"username": EMAIL, "password": PASSWORD})
    assert (r.status_code, r.json()["detail"]) == (400, "LOGIN_USER_NOT_VERIFIED")


def test_a_sign_up_with_a_14_character_password_is_refused(client):
    r = register(client, password="q" * 14)
    assert (r.status_code, r.json()["detail"]["code"]) == (400, "REGISTER_INVALID_PASSWORD")


def test_a_password_reset_signs_out_the_old_cookie(client, accounts, outbox):
    sign_in(client, accounts)
    token = client.cookies["__Host-stockintel"]
    client.post("/auth/forgot-password", json={"email": EMAIL})
    client.post("/auth/reset-password", json={"token": token_sent(outbox), "password": "rain on a tin roof at noon"})
    assert client.get("/probe", headers={"Cookie": f"__Host-stockintel={token}"}).status_code == 401


def test_a_password_reset_leaves_other_users_signed_in(client, accounts, outbox):
    sign_in(client, accounts, email="ravi@example.com")
    accounts.create_user(EMAIL, PASSWORD, verified=True)
    client.post("/auth/forgot-password", json={"email": EMAIL})
    reset = {"token": token_sent(outbox), "password": "rain on a tin roof at noon"}
    assert client.post("/auth/reset-password", json=reset).status_code == 200
    assert client.get("/probe").status_code == 200


@pytest.mark.parametrize("path, page", [("/auth/request-verify-token", "verify"), ("/auth/forgot-password", "reset")])
def test_emailed_links_use_the_public_url_whatever_host_the_request_names(client, accounts, outbox, path, page):
    accounts.create_user(EMAIL, PASSWORD, verified=False)
    outbox.sent.clear()
    client.post(path, json={"email": EMAIL}, headers={"Host": "evil.example"})
    assert link_sent(outbox).startswith(f"{PUBLIC_URL}/#/{page}?token=")


@pytest.mark.parametrize("ask, use, refusal", [("/auth/request-verify-token", "/auth/verify", "VERIFY_USER_BAD_TOKEN"),
                                               ("/auth/forgot-password", "/auth/reset-password",
                                                "RESET_PASSWORD_BAD_TOKEN")])
def test_an_emailed_token_does_not_work_under_another_secret(db_path, outbox, ask, use, refusal):
    signer = Accounts(db_path, PUBLIC_URL, SECRET, None, outbox)
    signer.create_user(EMAIL, PASSWORD, verified=False)
    client_for(signer).post(ask, json={"email": EMAIL})
    other = client_for(Accounts(db_path, PUBLIC_URL, "t" * 32, None, outbox))
    r = other.post(use, json={"token": token_sent(outbox), "password": "rain on a tin roof at noon"})
    assert (r.status_code, r.json()["detail"]) == (400, refusal)


def test_without_a_mailer_sign_up_confirm_and_reset_are_not_mounted(db_path):
    client = client_for(Accounts(db_path, PUBLIC_URL, SECRET, None, None))
    paths = ("/auth/register", "/auth/request-verify-token", "/auth/verify", "/auth/forgot-password",
             "/auth/reset-password")
    assert [client.post(path, json={}).status_code for path in paths] == [404] * len(paths)


def test_without_a_mailer_an_unconfirmed_user_is_refused_before_it_is_stored(db_path):
    with pytest.raises(RuntimeError, match="STOCKINTEL_SMTP_"):
        Accounts(db_path, PUBLIC_URL, SECRET, None, None).create_user(EMAIL, PASSWORD, verified=False)
    assert rows(db_path, "user") == 0


def test_google_sign_in_creates_a_verified_user(client):
    assert google_sign_in(client).status_code == 204
    assert client.get("/probe").status_code == 200


def test_google_sign_in_works_over_loopback_http(db_path, google):
    assert google_sign_in(client_for(Accounts(db_path, "http://127.0.0.1:8931", SECRET, google, None))).status_code == 204


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


def test_google_sign_in_replaces_an_unconfirmed_password_account_with_the_same_email(client, accounts, google, db_path):
    accounts.create_user(google.profile["email"], PASSWORD, verified=False)
    assert (google_sign_in(client).status_code, rows(db_path, "user")) == (204, 1)


def test_google_sign_in_keeps_its_own_account_even_when_unconfirmed(client, db_path):
    google_sign_in(client)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute('UPDATE "user" SET is_verified = 0')
        before = conn.execute('SELECT id FROM "user"').fetchall()
    google_sign_in(client)
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute('SELECT id FROM "user"').fetchall() == before


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
    for name in [*GOOD_ENV, "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", *SMTP_ENV]:
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
    ({**SMTP_ENV, "STOCKINTEL_SMTP_PASSWORD": None}, "STOCKINTEL_SMTP_PASSWORD"),
    ({**SMTP_ENV, "STOCKINTEL_SMTP_PORT": "smtps"}, "STOCKINTEL_SMTP_PORT"),
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


def test_from_env_with_every_smtp_setting_mounts_sign_up(monkeypatch, db_path):
    set_env(monkeypatch, {**GOOD_ENV, **SMTP_ENV})
    assert client_for(Accounts.from_env(db_path)).post("/auth/register", json={}).status_code == 422


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


@pytest.fixture
def smtp(monkeypatch):
    seen = {}

    class RecordingSmtp:
        def __init__(self, host, port, *, timeout, context):
            seen.update(server=(host, port), timeout=timeout, context=context)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, user, password):
            seen["login"] = (user, password)

        def send_message(self, message):
            seen["message"] = message

    set_env(monkeypatch, SMTP_ENV)
    monkeypatch.setattr(smtplib, "SMTP_SSL", RecordingSmtp)
    asyncio.run(mailer_from_env().send(EMAIL, "Confirm your email for StockIntel", "Open this link"))
    return seen


def test_the_mailer_checks_the_smtp_servers_certificate(smtp):
    assert (smtp["context"].verify_mode, smtp["context"].check_hostname) == (ssl.CERT_REQUIRED, True)


def test_the_mailer_sends_with_the_smtp_settings(smtp):
    assert (smtp["server"], smtp["login"], smtp["message"]["From"], smtp["message"]["To"]) == (
        ("smtp.example.com", 465), ("apikey", "smtp-password"), "StockIntel <noreply@example.com>", EMAIL)


def test_the_mailer_times_out_a_stuck_smtp_server(smtp):
    assert smtp["timeout"] == SMTP_TIMEOUT_S


def test_the_mailer_dates_each_email(smtp):
    assert abs(parsedate_to_datetime(smtp["message"]["Date"]) - datetime.now(timezone.utc)) < timedelta(minutes=1)


def test_the_mailer_gives_each_email_an_id_on_the_senders_domain(smtp):
    assert smtp["message"]["Message-ID"].endswith("@example.com>")
