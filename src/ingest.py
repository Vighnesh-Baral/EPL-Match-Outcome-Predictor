"""Data ingestion CLI.   python -m src.ingest <openfootball|football-data|statsbomb|fpl> [options]

Every command writes to Data/ and prints what it wrote.  Nothing is fabricated: sources that
are unreachable raise, and unavailable statistics stay missing (NaN).
"""
from __future__ import annotations

import argparse

from src.data_sources import football_data, fpl, openfootball, statsbomb


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="source", required=True)
    of = sub.add_parser("openfootball", help="download OpenFootball season files (CC0)")
    of.add_argument("--seasons", nargs="*", default=["2026-27"])
    fd = sub.add_parser("football-data", help="download Football-Data.co.uk seasons (results + odds)")
    fd.add_argument("--seasons", nargs="+", required=True)
    sb = sub.add_parser("statsbomb", help="aggregate Premier League 2015/16 xG from StatsBomb Open Data")
    sb.add_argument("--limit", type=int, default=None)
    sub.add_parser("fpl", help="snapshot the live FPL bootstrap-static player table")
    args = parser.parse_args(argv)

    if args.source == "openfootball":
        for season in args.seasons:
            print("wrote", openfootball.download_openfootball_season(season))
    elif args.source == "football-data":
        for season in args.seasons:
            print("wrote", football_data.download_season(season))
    elif args.source == "statsbomb":
        print("wrote", statsbomb.save(statsbomb.build_pl_2015_16_xg(limit=args.limit)))
        print(statsbomb.ATTRIBUTION)
    elif args.source == "fpl":
        from src.config import RAW_CURRENT_DIR
        frame = fpl.bootstrap_to_players(fpl.fetch_bootstrap())
        path = RAW_CURRENT_DIR / "fpl_bootstrap_snapshot.csv"
        frame.to_csv(path, index=False)
        print("wrote", path, len(frame), "players")


if __name__ == "__main__":
    main()
