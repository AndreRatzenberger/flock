import sys
from pathlib import Path

import uvicorn


def main():
    """Run the Flock web application."""
    # Ensure the webapp directory is in the Python path
    webapp_dir = Path(__file__).resolve().parent
    if str(webapp_dir) not in sys.path:
        sys.path.insert(0, str(webapp_dir.parent.parent))  # Add 'src' to path

    try:
        uvicorn.run(
            "flock.webapp.app.main:app",
            host="127.0.0.1",
            port=8008,
            reload=True,
        )
    except ModuleNotFoundError as e:
        print(f"Error loading webapp modules: {e}")
        print(
            "Make sure all required packages are installed and module structure is correct."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
