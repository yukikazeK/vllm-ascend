"""Install the prefix loader while retaining any existing deployment overlay."""
import importlib.machinery
import os
from pathlib import Path
import runpy
import sys

this = Path(__file__).resolve().parent
paths = [p for p in sys.path if Path(p).resolve() != this]
other = importlib.machinery.PathFinder.find_spec('sitecustomize', paths)
if other and other.origin:
    runpy.run_path(other.origin)

path = os.getenv('VLLM_LOCAL_PREFIX_CACHE_MANIFEST')
export = os.getenv('VLLM_LOCAL_PREFIX_CACHE_EXPORT')
if path or export:
    from local_prefix_loader import install
    install(manifest=Path(path) if path else None, export=Path(export) if export else None)
