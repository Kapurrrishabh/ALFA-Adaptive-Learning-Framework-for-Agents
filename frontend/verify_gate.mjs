/**
 * C8's gate, driven through a real browser: a question typed in the page returns a grounded answer with
 * its evidence shown.
 *
 * It types rather than posts. Every other check in this repository can reach the agent over HTTP, so the
 * only thing left worth measuring here is the part a request cannot reach -- that the page renders the
 * answer, the figures it was given, and the abstention when there is one.
 *
 * Needs both servers already running: `python3 scripts/serve.py` and `npm run dev`.
 *   node verify_gate.mjs [url]
 */

import { chromium } from "playwright";

const BASE = process.argv[2] || "http://localhost:3000";
const ACCOUNT = `browser+${Date.now()}@example.com`;
const PASSWORD = "a-long-enough-password";
const QUESTIONS = [
  "how has AAPL been doing over the last month ?",
  "is NVDA overbought right now ?",
  "how volatile is TSLA at the moment ?",
];

const browser = await chromium.launch();
const page = await browser.newPage();
const problems = [];
page.on("pageerror", (error) => problems.push(`page error: ${error.message}`));

try {
  await page.goto(`${BASE}/login`, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Create account" }).click();
  await page.locator('input[type="email"]').fill(ACCOUNT);
  await page.locator('input[type="password"]').fill(PASSWORD);
  await page.getByRole("button", { name: "Generate credentials" }).click();

  await page.waitForURL(`${BASE}/`, { timeout: 30_000 });
  // Anchored, because "disconnected" contains "connected" and would pass a substring match.
  await page.getByText(/^connected —/).waitFor({ timeout: 30_000 });
  console.log(`registered and connected as ${ACCOUNT}`);

  const model = await page.getByText(/^generator /).innerText();
  const cut = await page.getByText(/^answer cut /).innerText();
  console.log(`the page names what is answering: ${model}, ${cut}`);

  let spoke = 0;
  let withheld = 0;
  for (const question of QUESTIONS) {
    const box = page.locator('input[placeholder*="AAPL"]');
    await box.fill(question);
    await box.press("Enter");

    // Found by its own question rather than by position, so a reply still rendering cannot be read as this one.
    const reply = page
      .locator("div.space-y-2")
      .filter({ has: page.getByText(question, { exact: true }) })
      .locator("div.rounded-2xl");
    // The reply is complete once the turn has been stored: that frame carries the verdict and the cut.
    await reply.getByText(/judged (right|wrong) by|unjudged/).waitFor({ timeout: 180_000 });

    const facts = await reply.locator("div.bg-\\[\\#080808\\]").allInnerTexts();
    if (facts.length === 0) problems.push(`${question}: the evidence panel rendered no figures`);

    const abstained = await reply.getByText("Withheld —").count();
    const answer = abstained
      ? await reply.getByText("Withheld —").innerText()
      : await reply.locator("p").first().innerText();
    if (!answer.trim()) problems.push(`${question}: the reply rendered no text`);
    abstained ? (withheld += 1) : (spoke += 1);

    // A figure in a spoken answer has to appear in the evidence panel, which is the grounding claim.
    // Matched against the whole panel, names included: "its 14 day rsi is 52" names the period of a
    // field the panel heads `RSI 14`, so values alone would call a correct answer invented.
    if (!abstained) {
      const shown = facts.join(" ");
      const quoted = answer.match(/[-+]?\d[\d.,]*%?/g) || [];
      const invented = quoted.filter((figure) => !shown.includes(figure));
      if (invented.length) problems.push(`${question}: answer states ${invented.join(", ")}, not in the evidence`);
      // The agent runs this same comparison server-side; if it disagrees with mine, one of us is wrong.
      const flagged = await reply.getByText("figures not in the evidence").count();
      if (flagged !== 0) problems.push(`${question}: the page reports figures the evidence does not carry`);
      console.log(`  spoke, ${quoted.length} figures, all in the ${facts.length} evidence lines`);
    } else {
      console.log(`  ${answer.replace(/\s+/g, " ")}, over ${facts.length} evidence lines`);
    }
  }

  console.log(`\n${QUESTIONS.length} typed in the browser: ${spoke} answered, ${withheld} withheld`);
  if (problems.length) {
    console.error(`\nFAILED\n${problems.map((one) => `  - ${one}`).join("\n")}`);
    process.exitCode = 1;
  } else {
    console.log("PASSED: every reply rendered its answer and the evidence behind it");
  }
} finally {
  await browser.close();
}
