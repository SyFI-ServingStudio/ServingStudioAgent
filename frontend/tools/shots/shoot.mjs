// Headless-browser screenshots of the running VibeSim chat UI.
//
//   npm install            # once (installs playwright)
//   npm run browser        # once per machine (downloads Chromium into ~/.cache/ms-playwright)
//   node shoot.mjs                          # default: http://127.0.0.1:8799
//   APP_URL=http://127.0.0.1:8765 node shoot.mjs
//   CONV="Create a new file" node shoot.mjs # pick a sidebar conversation by title substring
//   OUT=out-x node shoot.mjs                # custom output dir
//
// Loads the app, optionally selects a conversation, waits for the role-timeline
// cards to render, then captures the full viewport plus each assistant turn as
// its own image. Exits non-zero on any page/console error, so it doubles as a
// DOM smoke test.

import { chromium } from "playwright";
import { mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";

const URL = process.env.APP_URL || process.argv[2] || "http://127.0.0.1:8799";
const CONV = process.env.CONV || process.argv[3] || "";
const outDir = resolve(process.env.OUT || "out");

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });
const shot = (target, name) => target.screenshot({ path: resolve(outDir, name) });

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1440, height: 1600 },
  deviceScaleFactor: 2,
});
const errors = [];
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => {
  if (m.type() === "error") errors.push("console.error: " + m.text());
});

console.log("→ goto", URL);
await page.goto(URL, { waitUntil: "networkidle" });
await page.waitForTimeout(500);

if (CONV) {
  const item = page.locator("aside button", { hasText: CONV }).first();
  if (await item.count()) {
    await item.click();
    await page.waitForTimeout(500);
  } else {
    errors.push(`no sidebar conversation matched "${CONV}"`);
  }
}

// Wait for the role-timeline cards to render (each card carries .animate-grow).
await page
  .waitForSelector(".animate-grow", { timeout: 10000 })
  .catch(() => errors.push("no timeline cards rendered"));
await page.waitForTimeout(400);

await shot(page, "00-app.png"); // full viewport: sidebar + timeline + composer

// Each assistant turn is one .animate-rise block; capture them at natural height.
const turns = page.locator("section.scroll .animate-rise");
const count = await turns.count();
for (let i = 0; i < count; i += 1) {
  await shot(turns.nth(i), `turn-${String(i + 1).padStart(2, "0")}.png`);
}

await browser.close();
console.log(`shots → ${outDir} (${count} turn image${count === 1 ? "" : "s"})`);
console.log("page errors:", errors.length ? "\n - " + errors.join("\n - ") : "none");
process.exit(errors.length ? 1 : 0);
