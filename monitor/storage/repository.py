from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, Iterable, List

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker, selectinload

from schema import ModerationResult, PipelineResult

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

    def save_results(self, run_id: int, results: Iterable[ModerationResult | PipelineResult]) -> None:
        with self.session() as session:
            for result in results:
                # Handle both legacy ModerationResult and new PipelineResult
                if isinstance(result, PipelineResult):
                    # New pipeline result with full stages
                    record = ModerationResultRecord(
                        run_id=run_id,
                        prompt_text=result.prompt.text,
                        prompt_metadata=result.prompt.metadata.model_dump(),
                        prompt_payload=result.prompt.model_dump(),
                        # Input classification
                        input_flagged=result.input_classification.flagged,
                        input_raw_response=result.input_classification.raw_response,
                        # Answer generation
                        answer_text=result.answer.text if result.answer else None,
                        answer_model=result.answer.model if result.answer else None,
                        answer_raw_response=result.answer.raw_response if result.answer else None,
                        # Output classification
                        output_flagged=result.output_classification.flagged if result.output_classification else None,
                        output_raw_response=result.output_classification.raw_response if result.output_classification else None,
                        # Legacy compatibility
                        flagged=result.input_classification.flagged,
                        raw_response=result.input_classification.raw_response,
                    )
                    session.add(record)
                    session.flush()

                    # Save input classification flags
                    for flag in result.input_classification.flags:
                        session.add(
                            ModerationFlagRecord(
                                result_id=record.id,
                                category=flag.category,
                                score=flag.score,
                                violated=flag.violated,
                                flag_type="input",
                            )
                        )
                    
                    # Save output classification flags
                    if result.output_classification:
                        for flag in result.output_classification.flags:
                            session.add(
                                ModerationFlagRecord(
                                    result_id=record.id,
                                    category=flag.category,
                                    score=flag.score,
                                    violated=flag.violated,
                                    flag_type="output",
                                )
                            )
                else:
                    # Legacy ModerationResult (backward compatibility)
                    # Check if result has the new structure or old structure
                    if hasattr(result, 'input_classification'):
                        # It's actually a PipelineResult being passed as ModerationResult
                        record = ModerationResultRecord(
                            run_id=run_id,
                            prompt_text=result.prompt.text,
                            prompt_metadata=result.prompt.metadata.model_dump(),
                            prompt_payload=result.prompt.model_dump(),
                            input_flagged=result.input_classification.flagged,
                            input_raw_response=result.input_classification.raw_response,
                            answer_text=result.answer.text if result.answer else None,
                            answer_model=result.answer.model if result.answer else None,
                            answer_raw_response=result.answer.raw_response if result.answer else None,
                            output_flagged=result.output_classification.flagged if result.output_classification else None,
                            output_raw_response=result.output_classification.raw_response if result.output_classification else None,
                            flagged=result.input_classification.flagged,
                            raw_response=result.input_classification.raw_response,
                        )
                        session.add(record)
                        session.flush()
                        for flag in result.input_classification.flags:
                            session.add(
                                ModerationFlagRecord(
                                    result_id=record.id,
                                    category=flag.category,
                                    score=flag.score,
                                    violated=flag.violated,
                                    flag_type="input",
                                )
                            )
                        if result.output_classification:
                            for flag in result.output_classification.flags:
                                session.add(
                                    ModerationFlagRecord(
                                        result_id=record.id,
                                        category=flag.category,
                                        score=flag.score,
                                        violated=flag.violated,
                                        flag_type="output",
                                    )
                                )
                        continue
                    else:
                        # True legacy ModerationResult
                        record = ModerationResultRecord(
                            run_id=run_id,
                            prompt_text=result.prompt.text,
                            prompt_metadata=result.prompt.metadata.model_dump(),
                            prompt_payload=result.prompt.model_dump(),
                            input_flagged=result.flagged,
                            input_raw_response=result.raw_response,
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
                                flag_type="input",
                            )
                        )

    def persist_log(self, run_id: int, level: str, message: str) -> None:
        with self.session() as session:
            session.add(ModerationLogRecord(run_id=run_id, level=level, message=message))

    def list_runs(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self.session() as session:
            stmt = (
                select(ModerationRun)
                .options(
                    selectinload(ModerationRun.results).selectinload(ModerationResultRecord.flags),
                    selectinload(ModerationRun.logs),
                )
                .order_by(ModerationRun.created_at.desc())
                .limit(limit)
            )
            runs = session.scalars(stmt).all()
            return [self._serialize_run(run) for run in runs]

    def fetch_run_details(self, run_id: int) -> Dict[str, Any] | None:
        with self.session() as session:
            stmt = (
                select(ModerationRun)
                .where(ModerationRun.id == run_id)
                .options(
                    selectinload(ModerationRun.results).selectinload(ModerationResultRecord.flags),
                    selectinload(ModerationRun.logs),
                )
            )
            run = session.scalars(stmt).first()
            if not run:
                return None
            return self._serialize_run(run, include_details=True)

    def record_human_review(self, result_id: int, label: str, notes: str | None = None) -> bool:
        with self.session() as session:
            record = session.get(ModerationResultRecord, result_id)
            if not record:
                logger.warning("Result %s not found when recording human review.", result_id)
                return False
            record.human_label = label
            record.human_notes = notes or None
            record.human_reviewed_at = datetime.utcnow()
            session.add(record)
            return True

    def _serialize_run(self, run: ModerationRun, include_details: bool = False) -> Dict[str, Any]:
        input_flagged_count = sum(1 for result in run.results if result.input_flagged)
        output_flagged_count = sum(1 for result in run.results if result.output_flagged)
        reviewed_count = sum(1 for result in run.results if result.human_label)
        answers_generated = sum(1 for result in run.results if result.answer_text)

        payload: Dict[str, Any] = {
            "id": run.id,
            "dataset_id": run.dataset_id,
            "dataset_split": run.dataset_split,
            "model": run.model,
            "prompt_limit": run.prompt_limit,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "status": run.status,
            "output_path": run.output_path,
            "extra_args": run.extra_args or {},
            "input_flagged_count": input_flagged_count,
            "output_flagged_count": output_flagged_count,
            "flagged_count": input_flagged_count,  # Legacy compatibility
            "reviewed_count": reviewed_count,
            "answers_generated": answers_generated,
        }

        if include_details:
            payload["results"] = [self._serialize_result(result) for result in run.results]
            payload["logs"] = [self._serialize_log(log) for log in run.logs]
        return payload

    def _serialize_result(self, result: ModerationResultRecord) -> Dict[str, Any]:
        # Separate input and output flags
        input_flags = [self._serialize_flag(flag) for flag in result.flags if flag.flag_type == "input"]
        output_flags = [self._serialize_flag(flag) for flag in result.flags if flag.flag_type == "output"]
        
        return {
            "id": result.id,
            "prompt_text": result.prompt_text,
            "prompt_metadata": result.prompt_metadata or {},
            "prompt_payload": result.prompt_payload or {},
            # Input classification
            "input_flagged": result.input_flagged,
            "input_raw_response": result.input_raw_response or {},
            "input_flags": input_flags,
            # Answer generation
            "answer_text": result.answer_text,
            "answer_model": result.answer_model,
            "answer_raw_response": result.answer_raw_response or {},
            # Output classification
            "output_flagged": result.output_flagged,
            "output_raw_response": result.output_raw_response or {},
            "output_flags": output_flags,
            # Human review
            "human_label": result.human_label,
            "human_label_type": result.human_label_type,
            "human_notes": result.human_notes,
            "human_reviewed_at": result.human_reviewed_at.isoformat() if result.human_reviewed_at else None,
            # Legacy compatibility
            "flagged": result.flagged,
            "raw_response": result.raw_response or {},
            "flags": input_flags,  # For backward compatibility
        }

    @staticmethod
    def _serialize_flag(flag: ModerationFlagRecord) -> Dict[str, Any]:
        return {
            "category": flag.category,
            "score": flag.score,
            "violated": flag.violated,
        }

    @staticmethod
    def _serialize_log(log: ModerationLogRecord) -> Dict[str, Any]:
        return {
            "level": log.level,
            "message": log.message,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }


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
