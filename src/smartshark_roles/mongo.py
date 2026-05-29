from __future__ import annotations

from typing import Any

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError


def get_client(config: dict[str, Any]) -> MongoClient:
    mongo_config = config.get("mongodb", {})
    return MongoClient(
        mongo_config.get("uri", "mongodb://localhost:27018"),
        serverSelectionTimeoutMS=int(mongo_config.get("server_selection_timeout_ms", 5000)),
        connectTimeoutMS=int(mongo_config.get("connect_timeout_ms", 5000)),
        socketTimeoutMS=int(mongo_config.get("socket_timeout_ms", 120000)),
        retryWrites=False,
        appname="smartshark_roles_readonly_pipeline",
    )


def get_database(config: dict[str, Any], logger: Any | None = None) -> Database:
    client = get_client(config)
    database_name = config.get("mongodb", {}).get("database", "smartshark_2_2")
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as exc:
        raise RuntimeError(
            "Cannot connect to MongoDB. Start MongoDB on the configured URI before running this step."
        ) from exc
    except PyMongoError as exc:
        raise RuntimeError("MongoDB ping failed.") from exc

    if logger is not None:
        logger.info("Connected to MongoDB database=%s", database_name)
    return client[database_name]
