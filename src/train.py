"""Train and save the production model.

    python -m src.train            # PitchIQ v2.5 production ensemble (default)
    python -m src.train --legacy   # retrain the archived v2.4 Random Forest + Poisson model

The v2.5 recipe and hyper-parameters are frozen (see reports/v25/); this command only refits them
on all complete historical seasons and primes ratings with the known 2026/27 results.
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true", help="retrain the archived v2.4 model instead")
    args = parser.parse_args()
    if args.legacy:
        from src.historical_model import train_and_save_model
        bundle = train_and_save_model()
        print(json.dumps(bundle["metadata"], indent=2, default=str)[:2000])
        return
    from src.data_loader import load_current_season_results, load_historical_matches
    from src.data_sources.openfootball import load_openfootball_season
    from src.engine.production import MODEL_FILE, known_current_results, train_production_engine
    current = known_current_results(load_openfootball_season("2026-27"), load_current_season_results())
    bundle = train_production_engine(load_historical_matches(), current)
    meta = bundle["metadata"]
    print(f"Saved {MODEL_FILE}")
    print(f"  version {meta['version']} | members {meta['ensemble_members']}")
    print(f"  supervised rows {meta['supervised_training_rows']} | state primed with {meta['state_includes_current_season_matches']} known 2026/27 results")


if __name__ == "__main__":
    main()
