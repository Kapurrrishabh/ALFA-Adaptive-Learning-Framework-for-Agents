"""Where the project keeps what is not code. Each location can be moved with an environment variable."""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("ALFA_ARTIFACTS", PACKAGE / "models" / "artifacts"))
DATA = Path(os.environ.get("ALFA_DATA", PACKAGE / "database" / "data"))
CONFIGS = PACKAGE / "models" / "configs"
FRONTEND = Path(os.environ.get("ALFA_FRONTEND", PACKAGE.parent.parent / "frontend"))
