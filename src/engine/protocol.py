"""Chronological evaluation protocol shared by tuning, selection and the final test.

  tuning window    : 2018-19 .. 2021-22  (rolling-origin; used to set hyper-parameters)
  selection window : 2022-23 .. 2023-24  (rolling-origin; used to choose the model set,
                                          ensemble weights and calibration)
  final test       : 2024-25 .. 2025-26  (touched exactly once, after everything is frozen)

2015-16 is a warm-up season: it initialises ratings/histories but is excluded from
supervised training and from every evaluation (cold-start rows are not representative).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

WARMUP_SEASONS = ("2015-16",)
TUNING_SEASONS = ("2018-19", "2019-20", "2020-21", "2021-22")
SELECTION_SEASONS = ("2022-23", "2023-24")
TEST_SEASONS = ("2024-25", "2025-26")
ALL_SEASONS = tuple(f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2026))


def season_start_year(season: str) -> int:
    return int(season[:4])


def training_rows(table: pd.DataFrame, eval_season: str, first_train_season: str = "2016-17") -> pd.Series:
    """Boolean mask of rows allowed in the training set for ``eval_season``."""
    return (table["Season"] >= first_train_season) & (table["Season"] < eval_season)


def recency_weights(table: pd.DataFrame, mask: pd.Series, eval_season: str, half_life_seasons: float | None) -> np.ndarray:
    """Exponential recency weights measured in seasons before ``eval_season`` (None => uniform)."""
    n = int(mask.sum())
    if half_life_seasons is None:
        return np.ones(n)
    age = season_start_year(eval_season) - table.loc[mask, "Season"].map(season_start_year).to_numpy()
    return 0.5 ** ((age - 1) / half_life_seasons)


def rolling_origin(table: pd.DataFrame, eval_seasons: tuple[str, ...], fit_predict: Callable[[pd.DataFrame, pd.DataFrame, str], np.ndarray],
                   first_train_season: str = "2016-17") -> pd.DataFrame:
    """Run ``fit_predict(train_rows, eval_rows, season) -> (n,3) probs`` for each season.

    Returns a frame with Date, Season, HomeTeam, AwayTeam, y and p_H/p_D/p_A for all
    evaluated matches.  The train rows contain only seasons strictly before ``season``.
    """
    out = []
    for season in eval_seasons:
        train = table[training_rows(table, season, first_train_season)]
        test = table[table["Season"] == season]
        if train.empty or test.empty:
            raise ValueError(f"No data to evaluate {season}.")
        if train["Date"].max() >= test["Date"].min():
            raise AssertionError("Training rows overlap the evaluation period (leakage).")
        probs = np.asarray(fit_predict(train, test, season), dtype=float)
        frame = test[["Date", "Season", "HomeTeam", "AwayTeam", "y"]].copy()
        frame[["p_H", "p_D", "p_A"]] = probs
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def probs_of(frame: pd.DataFrame) -> np.ndarray:
    return frame[["p_H", "p_D", "p_A"]].to_numpy(dtype=float)
