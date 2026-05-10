"""Auto-register domain packs for the full test suite.

Domain packages live in packages/ within the monorepo. In development they are
not pip-installed; instead we add them to sys.path here so pytest can find them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_packages = Path(__file__).parent.parent / "packages"
sys.path.insert(0, str(_packages / "claudium-finance"))
sys.path.insert(0, str(_packages / "claudium-legal"))

import claudium_finance  # noqa: E402, F401 — registers finance-audit domain
import claudium_legal  # noqa: E402, F401 — registers legal-compliance domain
