from __future__ import annotations

import subprocess
import sys


def _pipe_to(cmd: list[str], data: bytes) -> bool:
    """Pipe data to a command's stdin. Returns True if exit code is 0."""
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(data)
    return proc.returncode == 0


def copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    if sys.platform == "darwin":
        cmd = ["pbcopy"]
    elif sys.platform.startswith("linux"):
        cmd = ["xclip", "-selection", "clipboard"]
    elif sys.platform == "win32":
        cmd = ["clip"]
    else:
        return False

    try:
        return _pipe_to(cmd, text.encode())
    except FileNotFoundError:
        return False
