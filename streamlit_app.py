"""Root entrypoint for Streamlit Community Cloud.

Streamlit re-runs this file on every interaction. We execute the real app
module fresh each run instead of importing it: `from dashboard.app import *`
only renders the first time the module is imported, and Python caches the
import afterwards, leaving the page blank on every subsequent run.
"""
import runpy
from pathlib import Path

_APP = Path(__file__).parent / "dashboard" / "app.py"
runpy.run_path(str(_APP), run_name="__main__")
