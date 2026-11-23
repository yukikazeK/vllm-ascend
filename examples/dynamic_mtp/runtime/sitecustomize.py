# SPDX-License-Identifier: Apache-2.0
"""Load the release-matched worker and engine modules from the launch overlay."""

import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

_RUNTIME_DIR = Path(__file__).resolve().parent
_SEARCH_PATH = [path for path in sys.path if Path(path).resolve() != _RUNTIME_DIR]
_existing = importlib.machinery.PathFinder.find_spec("sitecustomize", _SEARCH_PATH)
if _existing and _existing.origin:
    runpy.run_path(_existing.origin)


class DynamicMTPFinder:
    def __init__(self, overlay):
        self.overlay = overlay
        self.modules = json.loads((overlay / "modules.json").read_text())

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.modules:
            return importlib.util.spec_from_file_location(fullname, self.overlay / f"{fullname}.py")
        return None


if overlay := os.environ.get("DYN_FULL_ROOT"):
    sys.meta_path.insert(0, DynamicMTPFinder(Path(overlay)))
