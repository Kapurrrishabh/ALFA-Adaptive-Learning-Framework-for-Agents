"""A real app server on 127.0.0.1 for the browser tests: synthetic prices, no outside calls, the test key."""
import socket
import threading
import time

import pandas as pd
import pytest

KEY = "k" * 24
STOCKS = pd.DataFrame({"symbol": ["TEST", "PEER"], "name": ["Test Industries Ltd.", "Peer Power Ltd."],
                       "sector": ["Industrials", "Energy"], "nifty200": [True, True]})


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    import uvicorn
    import backend.advisory.calculators.performance as perf
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
    # the same pages whether or not this checkout holds the built track record or model weights
    missing = tmp_path_factory.mktemp("not-built")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("STOCKINTEL_ML", "0")
        mp.delenv("STOCKINTEL_MODEL_SPACE", raising=False)
        mp.setenv("SELFAGENT_URL", "http://127.0.0.1:9")   # nothing listens on the discard port, so the agent reads as off
        mp.setattr(perf, "PATH", missing / "performance.json")
        mp.setattr(alfa_fan, "WEIGHTS", missing / "alfa.npz")
        mp.setattr(gru_line, "WEIGHTS", missing / "path_gru.npz")
        mp.setattr(insider, "load", no_insider)
        provider = SyntheticProvider()
        app = create_app(Orchestrator(provider, Store(":memory:"), use_rss_news=False), api_key=KEY)
        hub = app.state.hub
        hub._panel, hub._panel_at = panel_from_frames({s: provider.ohlcv(resolve_symbol(s)) for s in STOCKS["symbol"]}), time.time()
        hub._names = STOCKS
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("the browser tests' app server did not start on 127.0.0.1")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
        server.should_exit = True
        thread.join(timeout=10)
