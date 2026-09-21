import pytest

from selfagent import backend


@pytest.fixture(autouse=True, scope="session")
def fail_on_non_finite_values():
    """Tests run with numeric checking on, so an inf or nan surfaces at the op that made it."""
    backend.set_check_numerics(True)
    yield
    backend.set_check_numerics(False)
