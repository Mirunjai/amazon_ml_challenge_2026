#!/usr/bin/env python3
"""CLI placeholder for the final end-to-end pipeline.

The matching logic will be finalized after Source 2 and Source 3 are profiled.
Keep the notebook as the experiment driver; use this script once the final
pipeline functions are locked.
"""

from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--student-resource", required=True)
    parser.add_argument("--output-dir", default="outputs/final")
    args = parser.parse_args()

    root = Path(args.student_resource).expanduser().resolve()
    print(f"Student resource: {root}")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print("Pipeline CLI scaffold ready; final blocking/model steps are intentionally pending.")


if __name__ == "__main__":
    main()
