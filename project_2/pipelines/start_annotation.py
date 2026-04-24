"""
Project 2 - Annotation launcher
===============================

Runs the setup needed for manual annotation, then opens the GUI.

Usage:
  python project_2/start_annotation.py
"""

from __future__ import annotations

from apps.annotate_gui import main as annotate_main
from pipelines.build_manifest import main as build_manifest_main
from pipelines.export_roi_tasks import main as export_roi_tasks_main


def main() -> None:
    print("=" * 60)
    print("Project 2 annotation setup")
    print("=" * 60)

    print("\n[1/3] Building manifest...")
    build_manifest_main()

    print("\n[2/3] Syncing annotation sheet...")
    export_roi_tasks_main()

    print("\n[3/3] Launching annotation GUI...")
    annotate_main()


if __name__ == "__main__":
    main()
