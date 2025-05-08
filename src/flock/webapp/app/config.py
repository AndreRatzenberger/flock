import os
from pathlib import Path

FLOCK_FILES_DIR = Path(os.getenv("FLOCK_FILES_DIR", "./.flock_ui_projects"))
FLOCK_FILES_DIR.mkdir(parents=True, exist_ok=True)

# Global state for MVP - NOT SUITABLE FOR PRODUCTION/MULTI-USER
CURRENT_FLOCK_INSTANCE = None
CURRENT_FLOCK_FILENAME = None
