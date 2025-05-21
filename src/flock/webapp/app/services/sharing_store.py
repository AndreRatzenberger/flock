import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import aiosqlite

from flock.webapp.app.services.sharing_models import (
    FeedbackRecord,
    SharedLinkConfig,
)

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

    # Feedback
    @abstractmethod
    async def save_feedback(self, record: FeedbackRecord):
        """Persist a feedback record."""
        pass

class SQLiteSharedLinkStore(SharedLinkStoreInterface):
    """SQLite implementation for storing and retrieving shared link configurations."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        logger.info(f"SQLiteSharedLinkStore initialized with db_path: {self.db_path}")

    async def initialize(self) -> None:
        """Initializes the database and creates/updates the table if it doesn't exist."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Ensure the table exists with the base schema first
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS shared_links (
                        share_id TEXT PRIMARY KEY,
                        agent_name TEXT NOT NULL,
                        flock_definition TEXT NOT NULL,
                        created_at TEXT NOT NULL
                        /* New columns will be added below if they don't exist */
                    )
                    """
                )

                # Add new columns individually, ignoring errors if they already exist
                new_columns = [
                    ("share_type", "TEXT DEFAULT 'agent_run' NOT NULL"),
                    ("chat_message_key", "TEXT"),
                    ("chat_history_key", "TEXT"),
                    ("chat_response_key", "TEXT")
                ]

                for column_name, column_type in new_columns:
                    try:
                        await db.execute(f"ALTER TABLE shared_links ADD COLUMN {column_name} {column_type}")
                        logger.info(f"Added column '{column_name}' to shared_links table.")
                    except sqlite3.OperationalError as e:
                        if "duplicate column name" in str(e).lower():
                            logger.debug(f"Column '{column_name}' already exists in shared_links table.")
                        else:
                            raise # Re-raise if it's a different operational error

                # Feedback table
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback (
                        feedback_id TEXT PRIMARY KEY,
                        share_id TEXT,
                        context_type TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        expected_response TEXT,
                        actual_response TEXT,
                        flock_name TEXT,
                        agent_name TEXT,
                        flock_definition TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(share_id) REFERENCES shared_links(share_id)
                    )
                    """
                )

                await db.commit()
            logger.info(f"Database initialized and shared_links table schema ensured at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"SQLite error during initialization: {e}", exc_info=True)
            raise

    async def save_config(self, config: SharedLinkConfig) -> SharedLinkConfig:
        """Saves a shared link configuration to the SQLite database."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO shared_links (
                        share_id, agent_name, created_at, flock_definition, 
                        share_type, chat_message_key, chat_history_key, chat_response_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        config.share_id,
                        config.agent_name,
                        config.created_at.isoformat(),
                        config.flock_definition,
                        config.share_type,
                        config.chat_message_key,
                        config.chat_history_key,
                        config.chat_response_key,
                    ),
                )
                await db.commit()
            logger.info(f"Saved shared link config for ID: {config.share_id} with type: {config.share_type}")
            return config
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving config for ID {config.share_id}: {e}", exc_info=True)
            raise

    async def get_config(self, share_id: str) -> SharedLinkConfig | None:
        """Retrieves a shared link configuration from SQLite by its ID."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(
                    """SELECT 
                        share_id, agent_name, created_at, flock_definition, 
                        share_type, chat_message_key, chat_history_key, chat_response_key 
                    FROM shared_links WHERE share_id = ?""",
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
                    share_type=row[4],
                    chat_message_key=row[5],
                    chat_history_key=row[6],
                    chat_response_key=row[7],
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

    # ----------------------- Feedback methods -----------------------

    async def save_feedback(self, record: FeedbackRecord) -> FeedbackRecord:
        """Persist a feedback record to SQLite."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT INTO feedback (
                        feedback_id, share_id, context_type, reason,
                        expected_response, actual_response, flock_name, agent_name, flock_definition, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.feedback_id,
                        record.share_id,
                        record.context_type,
                        record.reason,
                        record.expected_response,
                        record.actual_response,
                        record.flock_name,
                        record.agent_name,
                        record.flock_definition,
                        record.created_at.isoformat(),
                    ),
                )
                await db.commit()
            logger.info(f"Saved feedback {record.feedback_id} (share={record.share_id})")
            return record
        except sqlite3.Error as e:
            logger.error(f"SQLite error saving feedback {record.feedback_id}: {e}", exc_info=True)
            raise
