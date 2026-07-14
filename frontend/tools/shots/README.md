# VibeSim UI screenshots (headless)

A small, self-contained dev utility for capturing the running VibeSim chat UI with a headless
Chromium — handy for reviewing the role-timeline redesign (card coloring, streaming layout,
token footers) without opening a browser by hand. Kept separate from the frontend app so
Playwright never lands in the app's dependency tree.

## Setup (once)

```bash
cd frontend/tools/shots
npm install            # installs playwright
npm run browser        # downloads Chromium into ~/.cache/ms-playwright (shared, reused)
```

## Use

Start the backend first (it serves the built frontend), then:

```bash
node shoot.mjs                                   # default: http://127.0.0.1:8799
APP_URL=http://127.0.0.1:8765 node shoot.mjs     # a different port
CONV="Create a new file" node shoot.mjs          # pick a conversation by sidebar-title substring
OUT=out-x node shoot.mjs                          # custom output dir
```

On load the app auto-selects the most recent conversation. Screenshots land in `out/`
(gitignored):

- `00-app.png` — the full viewport: sidebar, timeline, composer
- `turn-NN.png` — each assistant turn as its own full-height image (one per turn)

It exits non-zero on any page/console error, so it doubles as a DOM smoke test.
