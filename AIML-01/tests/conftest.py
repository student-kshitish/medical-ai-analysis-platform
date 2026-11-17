import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AIML_ROOT = PROJECT_ROOT / "AIML-01"

sys.path.insert(0, str(AIML_ROOT))
