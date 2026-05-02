#!/usr/bin/env python3
"""Launch the PySide6 pixel RGB editor."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from pixel_rgb_editor.tabbed_main_window import main as gui_main
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            print(
                "PySide6 is required for the GUI. Install it with:\n"
                "  python3 -m pip install -r requirements_pixel_rgb_editor.txt",
                file=sys.stderr,
            )
            return 1
        raise
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
