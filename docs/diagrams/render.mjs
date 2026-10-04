// Renders docs/diagrams/*.html to PNG next to docs/. Run from the repo root:
//   node docs/diagrams/render.mjs          (uses frontend/node_modules/playwright)
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(path.join(here, "../../frontend/package.json"));
const { chromium } = require("playwright");

const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 2 });
for (const name of ["architecture"]) {
  await page.goto(`file://${path.join(here, `${name}.html`)}`);
  const svg = await page.$("#diagram");
  await svg.screenshot({ path: path.join(here, "..", `${name}.png`) });
  console.log(`wrote docs/${name}.png`);
}
await browser.close();
