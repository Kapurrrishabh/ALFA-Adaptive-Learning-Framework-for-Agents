"""A real app server on 127.0.0.1 for the browser tests: synthetic prices, no outside calls, one user who
signs in through the login form, a small track record and one holding."""
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from tests.app.conftest import EMAIL, PASSWORD, SECRET

KEY = "k" * 24      # the web app asks for a key until its sign-in screens ship; the server ignores it
RUN_REQUESTS = 700   # provisional: keyed requests the pages sent in one full browser run (683 on 2026-11-01, 676 on 2026-11-02); recount when the suite grows
STOCKS = pd.DataFrame({"symbol": ["TEST", "PEER"], "name": ["Test Industries Ltd.", "Peer Power Ltd."],
                       "sector": ["Industrials", "Energy"], "nifty200": [True, True]})


def track_record(dates):
    """The fields the track record page and Home read, shaped as performance.build writes them."""
    weekly = [str(d.date()) for d in dates[::5]]
    growth = {"Momentum 20 (backtest)": 1.003, "Momentum + risk overlay (backtest)": 1.0025,
              "Equal-weight same stocks (backtest)": 1.002, "Nifty 500 price index, no dividends (real)": 1.0015}
    return {"built_at": "2026-10-01T00:00:00+00:00", "bonferroni_z": 3.25,
            "patterns": [{"family": "candlestick", "pattern": "hammer", "horizon": 5, "excess_pct": 0.3, "ci_lo": -0.4, "ci_hi": 1.0,
                          "z": 0.9, "events": 120, "stocks": 2, "verdict": "no proven edge"},
                         {"family": "chart", "pattern": "double top", "horizon": 20, "excess_pct": -1.2, "ci_lo": -1.9, "ci_hi": -0.5,
                          "z": -3.4, "events": 40, "stocks": 2, "verdict": "proven edge"}],
            "forecast": {"horizons": [{"horizon": 5, "n": 900, "accuracy": 0.52, "always_up_accuracy": 0.53}],
                         "coverage": [{"symbol": "TEST", "horizon": 10, "coverage": 0.78}, {"symbol": "PEER", "horizon": 10, "coverage": 0.81}],
                         "calibration": {"5": [{"predicted": 0.45, "actual": 0.48}, {"predicted": 0.55, "actual": 0.53}],
                                         "20": [{"predicted": 0.42, "actual": 0.51}, {"predicted": 0.58, "actual": 0.56}]}},
            "momentum": {"dates": weekly, "curves": {name: [round(g ** k, 4) for k in range(len(weekly))] for name, g in growth.items()},
                         "table": {"momentum_semiannual": {"max_dd_pct": -35.0}, "momentum_trend_vol_overlay": {"max_dd_pct": -21.0}},
                         "live_check": {"window": "2023-01-02 to 2026-09-30",
                                        "yearly": [{"year": 2024, "etf": 30.1, "nifty500": 15.2}, {"year": 2025, "etf": -5.5, "nifty500": 4.6}],
                                        "bars": [{"label": "Real momentum ETF", "cagr": 14.0, "real": True},
                                                 {"label": "Real Nifty 500 (price only, no dividends)", "cagr": 12.0, "real": True},
                                                 {"label": "Backtest momentum", "cagr": 20.0, "real": False},
                                                 {"label": "Backtest equal-weight", "cagr": 15.0, "real": False}]}},
            "insider": [{"test": "Promoter buys, all panel stocks", "excess_20d": 0.8},
                        {"test": "Same stocks on random dates (placebo)", "excess_20d": 0.1}]}


def sign_in(context, base_url):
    """Sign a browser context in through the login form; the page's requests then carry its cookie."""
    r = context.request.post(f"{base_url}/auth/cookie/login", form={"username": EMAIL, "password": PASSWORD})
    assert r.ok, r.text()


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    import uvicorn
    import backend.advisory.calculators.performance as perf
    import backend.api.app as app_module
    from backend.api.accounts import Accounts, owner_of
    from backend.api.app import create_app
    from backend.api.orchestrator import Orchestrator
    from backend.config import resolve_symbol
    from backend.database.sources import insider
    from backend.database.sources.panel import panel_from_frames
    from backend.database.sources.provider import DataUnavailable
    from backend.database.sources.synthetic import SyntheticProvider
    from backend.database.storage import Store
    from backend.models.serving import alfa_fan, gru_line

    def no_insider():
        raise DataUnavailable("no insider file in the browser tests")

    class Provider(SyntheticProvider):
        def intraday(self, symbol, period="5d", interval="15m"):
            """The last 25 daily bars, as the quarter-hours of the last session."""
            bars = self.ohlcv(symbol).iloc[-25:].copy()
            bars.index = pd.date_range(bars.index[-1] + pd.Timedelta("9h15min"), periods=len(bars), freq="15min")
            return bars
    # the same pages whether or not this checkout holds the built track record or model weights
    built, missing = tmp_path_factory.mktemp("built"), tmp_path_factory.mktemp("not-built")
    provider = Provider()
    prices = provider.ohlcv(resolve_symbol("TEST"))
    (built / "performance.json").write_text(json.dumps(track_record(prices.index)))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STOCKINTEL_ML", "0")
        mp.delenv("STOCKINTEL_MODEL_SPACE", raising=False)
        mp.setenv("SELFAGENT_URL", "http://127.0.0.1:9")   # nothing listens on the discard port, so the agent reads as off
        mp.setattr(perf, "PATH", built / "performance.json")
        mp.setattr(perf, "MODELS_PATH", missing / "model_evals.json")
        mp.setattr(alfa_fan, "WEIGHTS", missing / "alfa.npz")
        mp.setattr(gru_line, "WEIGHTS", missing / "path_gru.npz")
        mp.setattr(insider, "load", no_insider)
        # every browser tab signs in as the same user, and all of them from 127.0.0.1
        mp.setattr(app_module, "SIGN_INS_PER_MIN", 10_000)
        mp.setattr(app_module, "EMAIL_TRIES", 10_000)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        accounts = Accounts(str(tmp_path_factory.mktemp("accounts") / "stockintel.db"), url, SECRET, None, None)
        # in another thread: Playwright's sync API runs an event loop here, and create_user starts its own
        with ThreadPoolExecutor(1) as pool:
            user = pool.submit(accounts.create_user, EMAIL, PASSWORD, verified=True).result()
        mp.setattr(app_module, "RATE_PER_MIN", 20 * RUN_REQUESTS)   # every test shares one user's bucket: 10x a run's total, the pages' RUN_REQUESTS and as many again from the test of it
        orch = Orchestrator(provider, Store(":memory:"), use_rss_news=False, charges_model_calls=True)
        orch.save_portfolio(owner_of(user), 0.0, "moderate", "medium", [{"symbol": "TEST", "side": "BUY", "quantity": 10,
                                                         "price": float(prices["close"].iloc[-120]), "trade_date": str(prices.index[-120].date())}])
        app = create_app(orch, accounts)
        hub = app.state.hub
        hub._panel, hub._panel_at = panel_from_frames({s: provider.ohlcv(resolve_symbol(s)) for s in STOCKS["symbol"]}), time.time()
        hub._names = STOCKS
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("the browser tests' app server did not start on 127.0.0.1")
            time.sleep(0.05)
        yield url
        server.should_exit = True
        thread.join(timeout=10)
