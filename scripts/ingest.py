"""Ingestion pipeline — load documents, chunk, embed, and store in pgvector."""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.splitter import load_and_split
from src.vector_store import add_documents, get_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest documents into pgvector")
    parser.add_argument("--docs-dir", required=True, help="Path to documents folder")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before ingesting",
    )
    args = parser.parse_args()

    if args.reset:
        logger.info("Resetting collection...")
        store = get_store()
        store.drop_tables()
        store.create_tables_if_not_exists()
        logger.info("Collection reset complete")

    chunks = load_and_split(args.docs_dir)
    logger.info("Loaded %d chunks from %s", len(chunks), args.docs_dir)

    add_documents(chunks)
    logger.info("Done — %d chunks stored", len(chunks))


if __name__ == "__main__":
    main()
