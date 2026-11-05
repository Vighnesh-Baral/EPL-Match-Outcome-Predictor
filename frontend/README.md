# PitchIQ frontend — editorial redesign

React + Vite + Tailwind CSS v4 frontend for the existing FastAPI backend.

## What changed

- Added a scroll-led product landing page inspired by the supplied pastel/editorial reference.
- Reworked the workspace into a warm, restrained analytics interface using ivory, charcoal, sage and muted terracotta rather than a generic neon dashboard.
- Added purposeful interactions: sticky navigation, scroll reveals, hover states, responsive navigation, filters, team drill-down, player search, head-to-head queries and live match predictions.
- Connected all six workspace destinations to existing FastAPI endpoints. Player intelligence retrieves player lists from the existing `/api/teams/{team_name}` endpoint; it does not invent player metrics.
- Added clear historical-backtest caveats to avoid presenting historic metrics as guarantees for future fixtures.
- Club identifiers are text initials. No Premier League or club crests are included.

## Development

Terminal 1 (project root):

```powershell
python run.py
```

Terminal 2 (`frontend` folder):

```powershell
npm install
npm run dev
```

Open http://127.0.0.1:5173. Vite proxies `/api` requests to FastAPI at port 8000.

## Production build

From the `frontend` folder:

```powershell
npm run build
```

Vite writes the compiled site to `app/static`, which is served by FastAPI. Then run `python run.py` from the project root and open http://127.0.0.1:8000.

## Image credits and licence notes

The landing and workspace pages use football photography from Pexels by Franco Monsalvo:

- [Dynamic soccer game action with player in focus](https://www.pexels.com/photo/dynamic-soccer-game-action-with-player-in-focus-32285150/) — used in the hero and workspace banner.
- [Intense football match with players in action](https://www.pexels.com/photo/intense-football-match-with-players-in-action-32179167/) — used in the match and methodology sections.

Pexels states that its photos can be used for free, including commercially, without required attribution. Depicted people, trademarks and brands may still carry third-party rights; the design avoids club crests and does not imply endorsement. See the [Pexels licence](https://www.pexels.com/license/) and [Pexels help article on commercial use](https://help.pexels.com/hc/en-us/articles/360042295214-Can-I-use-the-photos-and-videos-for-a-commercial-project).
