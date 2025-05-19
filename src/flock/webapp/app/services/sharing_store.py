import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import aiosqlite

from flock.webapp.app.services.sharing_models import SharedLinkConfig

# Get a logger instance
logger = logging.getLogger(__name__)

class SharedLinkStoreInterface(ABC):
    """Interface for storing and retrieving shared link configurations."""

    @abstractmethod
    async def save_config(self, config: SharedLinkConfig) -> SharedLinkConfig:
        """Saves a shared link configuration.

        Args:
            config: The SharedLinkConfig object to save.

        Returns:
            The saved SharedLinkConfig object (could be updated, e.g., with DB ID).
        """
        pass

    @abstractmethod
    async def get_config(self, share_id: str) -> SharedLinkConfig | None:
        """Retrieves a shared link configuration by its ID.

        Args:
            share_id: The unique ID of the shared link.

        Returns:
            The SharedLinkConfig object if found, otherwise None.
        """
        pass

    async def initialize(self):
        """Initializes the store (e.g., creates database tables if they don't exist)."""
        pass # Default implementation does nothing, can be overridden

class SQLiteSharedLinkStore(SharedLinkStoreInterface):
    """SQLite implementation for storing and retrieving shared link configurations."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    async def initialize(self):
        """Initializes the database and creates the table if it doesn\'t exist."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS shared_links (
                        share_id TEXT PRIMARY KEY,
                        agent_name TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                await db.commit()
            logger.info(f"SQLite database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database at {self.db_path}: {e}")
            raise

    async def save_config(self, config: SharedLinkConfig) -> SharedLinkConfig:
        """Saves a shared link configuration to the SQLite database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO shared_links (share_id, agent_name, created_at) VALUES (?, ?, ?)",
                    (config.share_id, config.agent_name, config.created_at.isoformat()),
                )
                await db.commit()
            logger.info(f"Saved shared link config with ID: {config.share_id}")
            return config
        except aiosqlite.IntegrityError as e:
            logger.error(f"Integrity error saving config {config.share_id}: {e} - Share ID likely already exists.")
            # Depending on desired behavior, you might re-raise, or return None, or handle specific cases
            raise # Re-raising for now
        except Exception as e:
            logger.error(f"Failed to save shared link config {config.share_id}: {e}")
            raise

    async def get_config(self, share_id: str) -> SharedLinkConfig | None:
        """Retrieves a shared link configuration by its ID from the SQLite database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT share_id, agent_name, created_at FROM shared_links WHERE share_id = ?",
                    (share_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return SharedLinkConfig(
                            share_id=row[0],
                            agent_name=row[1],
                            created_at=datetime.fromisoformat(row[2]),
                        )
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve shared link config {share_id}: {e}")
            raise # Or return None, depending on how errors should propagate
