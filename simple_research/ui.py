"""UI support."""

from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn


def spinner(message: str) -> Progress:
    """Returns a progress bar with a message."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    )
    progress.add_task(message, total=None)
    return progress
