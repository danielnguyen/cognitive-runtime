from __future__ import annotations

import argparse
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


def main() -> int:
    from services.relationship_bootstrap import apply_bootstrap_seed

    parser = argparse.ArgumentParser(description="Bootstrap trusted relationship graph records.")
    parser.add_argument("seed_path", help="Path to the relationship seed YAML file.")
    parser.add_argument(
        "--owner-id",
        required=True,
        help="Owner ID to apply the bootstrap data for.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and simulate without writing.",
    )
    args = parser.parse_args()

    try:
        result = apply_bootstrap_seed(
            owner_id=args.owner_id,
            seed_path=args.seed_path,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"errors=1 reason={exc}", file=sys.stderr)
        return 1

    print(f"entities_upserted={result['entities_upserted']}")
    print(f"relationships_upserted={result['relationships_upserted']}")
    print(f"evidence_inserted={result['evidence_inserted']}")
    print(f"evidence_skipped={result['evidence_skipped']}")
    print("errors=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
