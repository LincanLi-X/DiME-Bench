"""Paper-oriented tables, LaTeX, figure inputs, and reproduction reports."""

from dimebench.reporting.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    build_leaderboard,
    leaderboard_from_runs,
    load_leaderboard_summary,
    render_leaderboard_html,
    render_leaderboard_markdown,
)
from dimebench.reporting.paper_reproduction import (
    PaperReproductionConfig,
    ReproductionReport,
    load_reproduction_config,
    reproduce_paper,
)
from dimebench.reporting.submission import (
    ResultSubmission,
    create_result_submission,
)
from dimebench.reporting.tables import (
    PaperCell,
    PaperRow,
    PaperTable,
    build_paper_tables,
)

__all__ = [
    "PaperCell",
    "Leaderboard",
    "LeaderboardEntry",
    "PaperReproductionConfig",
    "PaperRow",
    "PaperTable",
    "ReproductionReport",
    "ResultSubmission",
    "build_paper_tables",
    "build_leaderboard",
    "leaderboard_from_runs",
    "load_reproduction_config",
    "load_leaderboard_summary",
    "create_result_submission",
    "render_leaderboard_html",
    "render_leaderboard_markdown",
    "reproduce_paper",
]
