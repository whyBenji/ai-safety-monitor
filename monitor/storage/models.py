from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for SQLAlchemy models."""


class ModerationRun(Base):
    __tablename__ = "moderation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    dataset_id: Mapped[str] = mapped_column(String(255))
    dataset_split: Mapped[str] = mapped_column(String(255))
    model: Mapped[str] = mapped_column(String(255))
    prompt_limit: Mapped[int] = mapped_column(Integer)
    output_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="running")
    extra_args: Mapped[dict] = mapped_column(JSON, default=dict)

    results: Mapped[list[ModerationResultRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="selectin"
    )
    logs: Mapped[list[ModerationLogRecord]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="selectin"
    )


class ModerationResultRecord(Base):
    __tablename__ = "moderation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("moderation_runs.id", ondelete="CASCADE"), index=True)
    prompt_text: Mapped[str] = mapped_column(Text)
    prompt_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    prompt_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Input classification
    input_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    input_raw_response: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Answer generation
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    answer_raw_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Output classification
    output_flagged: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    output_raw_response: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    
    # Human review (can review input, output, or both)
    human_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    human_label_type: Mapped[str | None] = mapped_column(String(32), nullable=True)  # 'input', 'output', or 'both'
    human_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    human_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Legacy fields for backward compatibility (computed properties would be better, but SQLAlchemy needs columns)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")  # Maps to input_flagged
    raw_response: Mapped[dict] = mapped_column(JSON, default=dict)  # Maps to input_raw_response

    run: Mapped[ModerationRun] = relationship(back_populates="results")
    flags: Mapped[list[ModerationFlagRecord]] = relationship(
        back_populates="result", cascade="all, delete-orphan", lazy="selectin"
    )


class ModerationFlagRecord(Base):
    __tablename__ = "moderation_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("moderation_results.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(255))
    score: Mapped[float] = mapped_column(Float)
    violated: Mapped[bool] = mapped_column(Boolean, default=False)
    flag_type: Mapped[str] = mapped_column(String(32), default="input")  # 'input' or 'output'

    result: Mapped[ModerationResultRecord] = relationship(back_populates="flags")


class ModerationLogRecord(Base):
    __tablename__ = "moderation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("moderation_runs.id", ondelete="CASCADE"), index=True)
    level: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[ModerationRun] = relationship(back_populates="logs")
