import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import aiosqlite

from flock.webapp.app.services.sharing_models import SharedLinkConfig

# Get a logger instance
logger = logging.getLogger(__name__)

class SharedLinkStoreInterface(ABC):
    """Interface for storing and retrieving shared link configurations."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the store (e.g., create tables)."""
        pass

    @abstractmethod
    async def save_config(self, config: SharedLinkConfig) -> SharedLinkConfig:
        """Saves a shared link configuration."""
        pass

    @abstractmethod
    async def get_config(self, share_id: str) -> SharedLinkConfig | None:
        """Retrieves a shared link configuration by its ID."""
        pass

    @abstractmethod
    async def delete_config(self, share_id: str) -> bool:
        """Deletes a shared link configuration by its ID. Returns True if deleted, False otherwise."""
        pass

class SQLiteSharedLinkStore(SharedLinkStoreInterface):
    """SQLite implementation for storing and retrieving shared link configurations."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        logger.info(f"SQLiteSharedLinkStore initialized with db_path: {self.db_path}")

    async def initialize(self) -> None:
        """Initializes the database and creates the table if it doesn't exist."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS shared_links (
                        share_id TEXT PRIMARY KEY,
                        agent_name TEXT NOT NULL,
                        flock_definition TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                await db.commit()
            logger.info(f"Database initialized and shared_links table ensured at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error during initialization: {e}", exc_info=True)
            raise

    async def save_config(self, config: SharedLinkConfig) -> SharedLinkConfig:
        """Saves a shared link configuration to the SQLite database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT INTO shared_links (share_id, agent_name, created_at, flock_definition) VALUES (?, ?, ?, ?)",
                    (
                        config.share_id,
                        config.agent_name,
                        config.created_at.isoformat(),
                        config.flock_definition,
                    ),
                )
                await db.commit()
            logger.info(f"Saved shared link config for ID: {config.share_id}")
            return config
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving config for ID {config.share_id}: {e}", exc_info=True)
            raise

    async def get_config(self, share_id: str) -> SharedLinkConfig | None:
        """Retrieves a shared link configuration from SQLite by its ID."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    "SELECT share_id, agent_name, created_at, flock_definition FROM shared_links WHERE share_id = ?",
                    (share_id,)
                ) as cursor:
                    row = await cursor.fetchone()
            if row:
                logger.debug(f"Retrieved shared link config for ID: {share_id}")
                return SharedLinkConfig(
                    share_id=row[0],
                    agent_name=row[1],
                    created_at=row[2], # SQLite stores as TEXT, Pydantic will parse from ISO format
                    flock_definition=row[3],
                )
            logger.debug(f"No shared link config found for ID: {share_id}")
            return None
        except sqlite3.Error as e:
            logger.error(f"SQLite error retrieving config for ID {share_id}: {e}", exc_info=True)
            return None # Or raise, depending on desired error handling

    async def delete_config(self, share_id: str) -> bool:
        """Deletes a shared link configuration from SQLite by its ID."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                result = await db.execute("DELETE FROM shared_links WHERE share_id = ?", (share_id,))
                await db.commit()
                deleted_count = result.rowcount
            if deleted_count > 0:
                logger.info(f"Deleted shared link config for ID: {share_id}")
                return True
            logger.info(f"Attempted to delete non-existent shared link config for ID: {share_id}")
            return False
        except sqlite3.Error as e:
            logger.error(f"SQLite error deleting config for ID {share_id}: {e}", exc_info=True)
            return False # Or raise
