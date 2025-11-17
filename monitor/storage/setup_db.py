from __future__ import annotations

import argparse

from .repository import ModerationRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create database tables for the moderation service.")
    parser.add_argument(
        "--database-url",
        required=True,
        help="SQLAlchemy-compatible database URL, e.g. postgresql+psycopg://user:pass@host:5432/db",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = ModerationRepository(args.database_url)
    repo.create_schema()
    print("Database schema ensured for:", args.database_url)


if __name__ == "__main__":
    main()
