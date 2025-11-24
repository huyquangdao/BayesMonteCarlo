from pathlib import Path
from typing import Iterable, Union


def append_to_log(log_path: Union[str, Path], lines: Iterable[str]) -> Path:
    """
    Append one or more text lines to a log file, creating parent directories when needed.
    A trailing newline is added to each provided line so callers can pass raw strings.
    """
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as lf:
        for line in lines:
            lf.write(line.rstrip("\n") + "\n")
    return log_file


def read_log(log_path: Union[str, Path]) -> str:
    """
    Read the full contents of a log file. Returns an empty string when the file is missing.
    """
    log_file = Path(log_path)
    if not log_file.exists():
        return ""
    return log_file.read_text(encoding="utf-8")
