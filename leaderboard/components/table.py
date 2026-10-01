"""HTML table component shared by downstream site wrappers."""

from dimebench.reporting import Leaderboard, render_leaderboard_html


def render_table(leaderboard: Leaderboard) -> str:
    """Render the canonical accessible leaderboard table."""
    return render_leaderboard_html(leaderboard)
