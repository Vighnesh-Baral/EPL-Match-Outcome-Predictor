# PitchIQ UI redesign — Version 2.3

## Design direction

The new interface combines the supplied warm/pastel reference with selective interaction patterns from the dark immersive reference:

- Warm ivory canvas, charcoal typography, muted sage and restrained terracotta.
- Editorial sans-serif headings paired with a serif italic accent.
- Large football photography used as a visual anchor, not as a repeated decorative background.
- Sticky navigation, scroll-triggered section reveals, hover lift, responsive navigation and reduced-motion support.
- A product landing page is separate from the analytics workspace.

## Workspace/API wiring

| Screen | Backend source | What it displays |
|---|---|---|
| Overview | `GET /api/overview`, `GET /api/teams` | Season snapshot, team-strength index, points table, model summary |
| Match predictor | `POST /api/predict` | Home/draw/away probabilities, expected goals, team component comparison and blend composition |
| Team intelligence | `GET /api/teams/{team_name}` | Club ratings, league snapshot and squad summary |
| Player intelligence | `GET /api/teams/{team_name}` for each club | Searchable player snapshot using returned player name, club, position, FPL points, contribution rating and cost |
| Head to head | `GET /api/h2h` | Historical meeting results and W/D/L summary from the selected club's perspective |
| Model performance | `GET /api/overview` | Historical holdout accuracy, log loss and Brier score |

The frontend uses the Vite proxy already configured in `frontend/vite.config.js`: `/api` is forwarded to `http://127.0.0.1:8000`.

## Visual/IP decisions

- Club marks are initials generated from club names. No official league or club crests are included.
- Football photos are attributed in the interface and in `ASSET_CREDITS.md`.
- Pexels images are requested from the Pexels CDN at runtime. Internet access is required to display them. Pexels' licence permits free use, including commercial use, while third-party rights in people, trademarks and brands depicted in a photo remain relevant.

## Verification completed

- Python test suite: **10 passed**.
- API smoke checks: all **20 club profiles** returned successfully; player profiles aggregated to **670 player records**; H2H and prediction endpoints returned successfully.
- Prediction smoke check: returned outcome probabilities summed to **100%**.
- JSX/JavaScript parser check: passed.
- CSS brace-balance check: passed.

## Verification not completed

A Vite production build was not completed in the packaging environment because `npm install` timed out while attempting to resolve/download dependencies. Run `npm install` in `frontend/` on a machine with npm registry access, then `npm run build` to verify the full production bundle. For development, run the backend and Vite frontend in separate terminals as documented in `README.md`.
