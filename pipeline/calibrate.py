"""Print the walk-forward calibration report.

Run with:  python pipeline/calibrate.py [WC EC ...]

With no arguments it back-tests every finished match; pass competition codes to
restrict the sample (e.g. ``python pipeline/calibrate.py WC``).
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db import get_session, init_db
from models.calibration import walk_forward_backtest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    competitions = sys.argv[1:] or None
    init_db()
    with get_session() as session:
        report = walk_forward_backtest(session, competitions=competitions)

    scope = ", ".join(competitions) if competitions else "all competitions"
    print(f"\nWalk-forward calibration — {scope}")
    print(f"  matches evaluated : {report['n_matches']}")
    if not report["n_matches"]:
        print("  (no finished matches to evaluate)")
        return
    print(f"  Brier score       : {report['brier']:.4f}   (lower is better)")
    print(f"  Log loss          : {report['log_loss']:.4f}   (lower is better)")
    print(f"  1X2 accuracy      : {report['accuracy']:.1%}")
    print(f"  Avg points/match  : {report['avg_points']:.3f}")
    print("\n  Reliability (predicted vs empirical, pooled over 1X2):")
    print("    bin   pred    emp    n")
    for b in report["reliability"]:
        if b["count"]:
            print(
                f"    {b['bin']:>3}  {b['mean_predicted']:.3f}  "
                f"{b['empirical_freq']:.3f}  {b['count']:>4}"
            )


if __name__ == "__main__":
    main()
