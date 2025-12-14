from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from monitor.storage import ModerationRepository

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def create_dashboard_app(database_url: str | None = None) -> FastAPI:
    db_url = database_url or os.environ.get("AI_SAFETY_MONITOR_DB", "sqlite:///./ai_monitor.db")
    repository = ModerationRepository(db_url)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app = FastAPI(title="AI Safety Monitor Dashboard")

    def _compute_metrics(runs: List[Dict]) -> Dict[str, int]:
        return {
            "total_runs": len(runs),
            "prompts_processed": sum(run.get("prompt_limit", 0) for run in runs),
            "input_flagged": sum(run.get("input_flagged_count", 0) for run in runs),
            "output_flagged": sum(run.get("output_flagged_count", 0) for run in runs),
            "answers_generated": sum(run.get("answers_generated", 0) for run in runs),
            "reviewed": sum(run.get("reviewed_count", 0) for run in runs),
            # Legacy compatibility
            "flagged": sum(run.get("flagged_count", 0) for run in runs),
        }

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        runs = repository.list_runs(limit=50)
        metrics = _compute_metrics(runs)
        context = {"request": request, "runs": runs, "metrics": metrics}
        return templates.TemplateResponse("index.html", context)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: int, only_flagged: int = 0) -> HTMLResponse:
        run = repository.fetch_run_details(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found.")

        results = run.get("results", [])
        if only_flagged:
            # Show results where either input or output is flagged
            results = [
                result
                for result in results
                if result.get("input_flagged") or result.get("output_flagged")
            ]

        input_flagged_total = sum(1 for result in run.get("results", []) if result.get("input_flagged"))
        output_flagged_total = sum(1 for result in run.get("results", []) if result.get("output_flagged"))
        reviewed_total = sum(1 for result in run.get("results", []) if result.get("human_label"))
        answers_generated = sum(1 for result in run.get("results", []) if result.get("answer_text"))

        context = {
            "request": request,
            "run": run,
            "results": results,
            "only_flagged": bool(only_flagged),
            "stats": {
                "input_flagged_total": input_flagged_total,
                "output_flagged_total": output_flagged_total,
                "reviewed_total": reviewed_total,
                "pending_reviews": input_flagged_total + output_flagged_total - reviewed_total,
                "answers_generated": answers_generated,
                # Legacy compatibility
                "flagged_total": input_flagged_total,
            },
        }
        return templates.TemplateResponse("run_detail.html", context)

    @app.post("/runs/{run_id}/results/{result_id}/review")
    async def review_result(
        request: Request,
        run_id: int,
        result_id: int,
        label: str = Form(...),
        notes: str = Form(""),
    ) -> RedirectResponse:
        success = repository.record_human_review(result_id, label.upper(), notes.strip() or None)
        if not success:
            raise HTTPException(status_code=404, detail="Result not found.")

        referer = request.headers.get("referer")
        redirect_url = referer or f"/runs/{run_id}"
        return RedirectResponse(url=redirect_url, status_code=303)

    @app.get("/health")
    async def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    return app


app = create_dashboard_app()
