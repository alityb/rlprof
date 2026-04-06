from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    package_root = Path(__file__).resolve().parent
    binary = package_root / "bin" / "hotpath"
    if not binary.exists():
        raise SystemExit(f"missing hotpath binary at {binary}")

    args = sys.argv[1:]
    command = args[0] if args else ""

    # ── rich serve-report renderer ───────────────────────────────────────────
    if command == "serve-report":
        db_path = next((a for a in args[1:] if not a.startswith("--")), None)
        use_text = "--format" in args and args[args.index("--format") + 1] == "text"

        if db_path and not use_text:
            try:
                from hotpath_py.serve_report import DEPS_AVAILABLE, render
                if DEPS_AVAILABLE:
                    sys.exit(render(db_path))
            except Exception:
                pass  # fall through to C++ binary on any error

    # ── default: hand off to C++ binary ─────────────────────────────────────
    env = os.environ.copy()
    env.setdefault("RLPROF_PYTHON_EXECUTABLE", sys.executable)
    os.execve(str(binary), [str(binary), *args], env)
