from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Iterable

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from schema import ModerationResult

from .models import (
    Base,
    ModerationFlagRecord,
    ModerationLogRecord,
    ModerationResultRecord,
    ModerationRun,
)

logger = logging.getLogger(__name__)


class ModerationRepository:
    """Manages persistence of moderation runs, results, flags, and logs."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url, future=True)
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:  # pragma: no cover - passthrough for CLI
            session.rollback()
            raise
        finally:
            session.close()

    def start_run(
        self,
        *,
        dataset_id: str,
        dataset_split: str,
        model: str,
        prompt_limit: int,
        output_path: str | None,
        extra_args: dict,
    ) -> ModerationRun:
        with self.session() as session:
            run = ModerationRun(
                dataset_id=dataset_id,
                dataset_split=dataset_split,
                model=model,
                prompt_limit=prompt_limit,
                output_path=output_path,
                extra_args=extra_args,
            )
            session.add(run)
            session.flush()
            session.refresh(run)
            return run

    def complete_run(self, run_id: int, status: str = "completed") -> None:
        with self.session() as session:
            run = session.get(ModerationRun, run_id)
            if not run:
                logger.warning("Run %s not found when attempting to complete.", run_id)
                return
            run.status = status
            run.completed_at = datetime.utcnow()
            session.add(run)

    def save_results(self, run_id: int, results: Iterable[ModerationResult]) -> None:
        with self.session() as session:
            for result in results:
                record = ModerationResultRecord(
                    run_id=run_id,
                    prompt_text=result.prompt.text,
                    prompt_metadata=result.prompt.metadata.model_dump(),
                    flagged=result.flagged,
                    raw_response=result.raw_response,
                )
                session.add(record)
                session.flush()

                for flag in result.flags:
                    session.add(
                        ModerationFlagRecord(
                            result_id=record.id,
                            category=flag.category,
                            score=flag.score,
                            violated=flag.violated,
                        )
                    )

    def persist_log(self, run_id: int, level: str, message: str) -> None:
        with self.session() as session:
            session.add(ModerationLogRecord(run_id=run_id, level=level, message=message))


class DatabaseLogHandler(logging.Handler):
    """Logging handler that forwards log records into the database."""

    def __init__(self, repository: ModerationRepository, run_id: int) -> None:
        super().__init__()
        self._repository = repository
        self._run_id = run_id

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - side-effect only
        try:
            msg = self.format(record)
            self._repository.persist_log(self._run_id, record.levelname, msg)
        except Exception:  # Never raise inside logging handler
            logger.exception("Failed to persist log record.")
