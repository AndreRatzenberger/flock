import os
from pathlib import Path

from flock.core import Flock  # Add type hint
from flock.core.logging.formatters.themes import OutputTheme

FLOCK_FILES_DIR = Path(os.getenv("FLOCK_FILES_DIR", "./.flock_ui_projects"))
FLOCK_FILES_DIR.mkdir(parents=True, exist_ok=True)

# Global state for MVP - NOT SUITABLE FOR PRODUCTION/MULTI-USER
CURRENT_FLOCK_INSTANCE: Flock | None = None
CURRENT_FLOCK_FILENAME: str | None = None

# Current Theme State
# Use a default theme from the OutputTheme enum if desired, or hardcode a string
DEFAULT_THEME_NAME = OutputTheme.afterglow.value
# Initialize CURRENT_THEME_NAME by reading the environment variable
_initial_theme_from_env = os.environ.get("FLOCK_WEB_THEME")
if _initial_theme_from_env and _initial_theme_from_env in [
    t.value for t in OutputTheme
]:
    CURRENT_THEME_NAME: str = _initial_theme_from_env
    print(
        f"Config: Initial theme set from FLOCK_WEB_THEME env var: {CURRENT_THEME_NAME}"
    )
else:
    if _initial_theme_from_env:
        print(
            f"Warning: Invalid theme name '{_initial_theme_from_env}' in FLOCK_WEB_THEME. Using default."
        )
    CURRENT_THEME_NAME: str = DEFAULT_THEME_NAME


def set_current_theme_name(theme_name: str | None):
    """Sets the globally accessible current theme name (used by integrated server)."""
    global CURRENT_THEME_NAME
    if theme_name and theme_name in [t.value for t in OutputTheme]:
        CURRENT_THEME_NAME = theme_name
        print(f"Set current theme to: {CURRENT_THEME_NAME}")
    else:
        # When called programmatically (integrated server), if theme is invalid/None, use default.
        if theme_name:
            print(
                f"Warning: Invalid theme name provided ('{theme_name}'). Using default: {DEFAULT_THEME_NAME}"
            )
        CURRENT_THEME_NAME = DEFAULT_THEME_NAME


def get_current_theme_name() -> str:
    """Gets the globally accessible current theme name."""
    # This will reflect the value set by env var (on module load) or by set_current_theme_name
    return CURRENT_THEME_NAME
