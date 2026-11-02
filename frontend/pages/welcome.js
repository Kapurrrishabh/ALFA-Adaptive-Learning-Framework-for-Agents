"use strict";
/* The setup wizard (#/welcome) and the signed-in summary at the top of Home. A new user has no
   portfolio row until step 1 is saved, so "not set up" is a 404 from /portfolio, never a flag. */

const RISK_WORDS = { conservative: "Careful", moderate: "Balanced", aggressive: "Bold" };
const MODE_WORDS = { defensive: "Your plan holds cash when the market trend is down, and less in stocks when prices swing hard.",
  balanced: "Your plan invests less when a momentum crash looks likely.", full: "Your plan stays fully invested in every market." };
const hasPage = (path) => ROUTES.some((r) => r.path === path);
const portfolioOrNone = (ctx) => api("/portfolio", { signal: ctx.signal }).catch((e) => { if (e.status === 404) return null; throw e; });

function choice(name, value, title, text, checked) {
  return h("label", { class: "card", style: "display:block;cursor:pointer" }, h("input", { type: "radio", name, value, checked }), " ", h("b", {}, title),
    text ? h("p", { class: "muted", style: "margin:6px 0 0" }, text) : null);
}

function aboutYou(ctx, p, profiles) {
  const fieldset = (legend, ...kids) => h("fieldset", { class: "grid cols-3", style: "border:0;padding:0;margin:0" }, h("legend", {}, h("h3", {}, legend)), ...kids);
  const cash = h("input", { type: "number", min: "0", step: "1", name: "cash", value: p ? String(p.cash) : "" });
  const problem = h("p", { class: "field-error", role: "alert" }), next = h("button", { class: "btn primary", type: "submit" }, "Next");
  const form = h("form", { class: "form", novalidate: true, onsubmit: async (e) => {
    e.preventDefault(); problem.textContent = "";
    const risk = form.elements.risk_profile.value, horizon = form.elements.horizon.value, money = cash.value.trim() ? Number(cash.value) : NaN;
    const missing = [!risk && "pick a risk level", !horizon && "pick how long you plan to hold", !(money >= 0) && "enter your cash in rupees (0 is fine)"].filter(Boolean);
    if (missing.length) { problem.textContent = `Please ${missing.join(", ")}.`; return; }
    next.disabled = true;
    try { await api("/portfolio/cash", { method: "POST", body: JSON.stringify({ cash: money, risk_profile: risk, horizon }) }); }
    catch (err) { next.disabled = false; problem.textContent = errorWords(err); return; }
    if (ctx.live()) location.hash = "#/welcome?step=2";
  } },
    fieldset("How much risk can you take?", ...Object.entries(profiles.risk).map(([name, r]) => choice("risk_profile", name, `${RISK_WORDS[name]} (${name})`,
      `No single stock above ${num(r.max_position_pct, 0)}% of your money, and no sector above ${num(r.max_sector_pct, 0)}%. ${MODE_WORDS[r.mode]}`, p && p.risk_profile === name))),
    fieldset("How long do you plan to hold?", ...Object.entries(profiles.horizon).map(([name, text]) => choice("horizon", name,
      `${name[0].toUpperCase()}${name.slice(1)}: ${text}`, null, p && p.horizon === name))),
    Field("Cash you can invest (₹)", cash, "Money you have set aside for stocks. 0 is fine."), problem, h("div", { class: "row" }, next));
  return [h("h2", {}, "About you"), form];
}

async function starterBasket(host, ctx, p, profiles) {
  host.replaceChildren(h("div", { class: "card" }, h("p", { class: "muted" }, "Picking stocks…"), skel(160)));
  let b; try { b = await api(`/ui/builder?capital=${p.cash}&risk=${profiles.risk[p.risk_profile].mode}`, { signal: ctx.signal }); }
  catch (e) { host.replaceChildren(ErrorState(e, () => starterBasket(host, ctx, p, profiles))); return; }
  if (!b.holdings.length) { host.replaceChildren(EmptyState("layers", "No stock to buy with your answers today", b.warnings.join(" ") || null, null)); return; }
  const prices = b.holdings.map((x) => h("input", { type: "number", min: "0.01", step: "0.01", value: x.price.toFixed(2) }));
  const problem = h("p", { class: "field-error", role: "alert" }), add = h("button", { class: "btn primary", type: "submit" }, "I've bought them, add them");
  host.replaceChildren(h("form", { class: "card form", novalidate: true, onsubmit: async (e) => {
    e.preventDefault(); problem.textContent = "";
    if (prices.some((input) => !(Number(input.value) > 0))) { problem.textContent = "Type the price you paid for each stock."; return; }
    add.disabled = true;
    try { await api("/ui/builder/apply", { method: "POST", body: JSON.stringify({ holdings: b.holdings.map((x, i) => ({ symbol: x.symbol, shares: x.shares, price: Number(prices[i].value) })) }) }); }
    catch (err) { add.disabled = false; problem.textContent = errorWords(err); return; }
    if (ctx.live()) location.hash = "#/welcome?step=3";
  } },
    h("h3", {}, "Your starter basket"),
    h("p", {}, `Buy these in your broker app, then come back. Change each price to the one you paid, so your portfolio holds what you really bought.`),
    ...b.holdings.map((x, i) => Field(`${x.name}: ${x.shares} shares, price paid per share (₹)`, prices[i])),
    b.warnings.length ? h("ul", { class: "checks con" }, ...b.warnings.map((w) => h("li", {}, w))) : null, problem, h("div", { class: "row" }, add)));
}

function addHoldings(ctx, p, profiles) {
  const count = p.transactions.length, min = profiles.builder.min_capital, basket = h("div");
  const tile = (title, text, action) => h("div", { class: "card" }, h("h3", {}, title), h("p", { class: "muted" }, text), action);
  return [h("h2", {}, "Add what you own"),
    count ? h("p", {}, `You have added ${count} transaction${count === 1 ? "" : "s"}.`) : null,
    h("div", { class: "grid cols-3" },
      tile("By hand", "Type in each stock you own: the shares, the price you paid and the date.",
        h("button", { class: "btn primary", onclick: () => addTxModal(ctx, "", null, route) }, count ? "Add another" : "Add a stock")),
      tile("Build a starter basket", `A basket for your ${inr(p.cash, 0)} of cash, picked to suit your answers.`, p.cash >= min
        ? h("button", { class: "btn", onclick: () => starterBasket(basket, ctx, p, profiles) }, "Show me a basket")
        : h("p", {}, `This needs at least ${inr(min, 0)} of cash. Go back to change your cash.`)),
      hasPage("import") ? tile("Import a file", "Read your holdings from your broker's file.", h("a", { class: "btn", href: "#/import" }, "Import a file")) : null),
    basket,
    h("div", { class: "row", style: "margin-top:16px" }, h("a", { class: "btn", href: "#/welcome?step=3" }, count ? "Next" : "I don't own any yet"))];
}

async function setupDone(ctx, p) {
  const rep = p.transactions.length ? await api("/portfolio/analysis", { signal: ctx.signal }) : null, first = rep && rep.holdings[0];
  return [h("h2", {}, "You're set up"),
    h("p", {}, rep ? `You hold ${rep.holdings.length} stock${rep.holdings.length === 1 ? "" : "s"} worth ${inr(rep.invested_value, 0)}, and ${inr(rep.cash, 0)} in cash.`
      : `You hold no stocks yet, and have ${inr(p.cash, 0)} in cash.`),
    h("div", { class: "row" }, h("a", { class: "btn primary", href: "#/signals" }, "See your plan"),
      first ? h("a", { class: "btn", href: `#/stock/${encodeURIComponent(first.symbol)}` }, `Open ${first.symbol}`) : h("a", { class: "btn", href: "#/explore" }, "Find a stock"),
      hasPage("guide") ? h("a", { class: "btn", href: "#/guide" }, "Read the guide") : null,
      h("a", { class: "btn", href: "#/home" }, "Go to Home"))];
}

async function pageWelcome(view, ctx) {
  view.append(h("div", { class: "card" }, skel(160)));
  const [p, profiles] = await Promise.all([portfolioOrNone(ctx), api("/ui/profiles", { signal: ctx.signal })]);
  const asked = Number(ctx.params.get("step")), step = p && [2, 3].includes(asked) ? asked : 1;
  const body = await [aboutYou, addHoldings, setupDone][step - 1](ctx, p, profiles);
  view.replaceChildren(h("div", { class: "card" }, h("div", { class: "spread" }, h("span", { class: "muted" }, `Step ${step} of 3`),
    step > 1 ? h("a", { class: "btn small", href: `#/welcome?step=${step - 1}` }, "Back") : null), ...body));
}

// Called by Home once signed in. Only the first Home of a tab sends a new user to setup, so they can still look around first.
async function homeSummary(view, ctx) {
  let p; try { p = await portfolioOrNone(ctx); } catch (e) { view.prepend(ErrorState(e, route)); return; }
  if (p === null) {
    if (!sessionStorage.getItem("si-welcomed")) { sessionStorage.setItem("si-welcomed", "1"); if (ctx.live()) location.hash = "#/welcome"; return; }
    view.prepend(h("div", { class: "card" }, h("h2", {}, "Finish setting up"),
      h("p", {}, "Tell us how much risk you can take, how long you plan to hold and your cash, so your plan and the chat fit you."),
      h("a", { class: "btn primary", href: "#/welcome" }, "Continue setup")));
    return;
  }
  const stat = (k, v, c = "") => h("div", { class: "stat" }, h("div", { class: "k" }, k), h("div", { class: `v num ${c}` }, v));
  const stats = h("div", { class: "grid cols-4" }, skel(48)), mood = h("p", {}, skel(16)), left = h("p", { class: "muted" }, skel(16));
  view.prepend(h("div", { class: "card" }, h("h2", {}, "Your money today"), stats, mood, left));
  const failed = (slot) => (e) => slot.replaceChildren(h("span", { class: "muted" }, errorWords(e)));
  if (p.transactions.length) {
    api("/portfolio/analysis", { signal: ctx.signal }).then((rep) => stats.replaceChildren(stat("Value", inr(rep.total_value, 0)),
      stat("Total gain", `${inr(rep.unrealized_pl, 0)} (${pct(rep.unrealized_pl_pct)})`, cls(rep.unrealized_pl)),
      stat("Last 5 sessions", `${inr(rep.week_change.total_change_value, 0)} (${pct(rep.week_change.total_change_pct)})`, cls(rep.week_change.total_change_value)),
      stat("Cash", inr(rep.cash, 0)))).catch(failed(stats));
  } else {
    stats.replaceChildren(stat("Cash", inr(p.cash, 0)), h("div", { class: "stat" }, h("div", { class: "k" }, "Holdings"), h("a", { href: "#/welcome?step=2" }, "Add what you own")));
  }
  api("/market", { signal: ctx.signal }).then((m) => { const up = m.trend === "up";
    mood.replaceChildren(`Market mood: the Nifty is in ${up ? "an uptrend" : "a downtrend"}. Its last month-end close of `
      + `${num(m.month_end_close, 0)} is ${up ? "above" : "below"} its 10-month average of ${num(m.sma_10m, 0)} (prices up to ${m.as_of}).`); }).catch(failed(mood));
  api("/ui/usage", { signal: ctx.signal }).then((u) => left.replaceChildren(`${u.limit - u.used} of ${u.limit} AI answers left today.`)).catch(failed(left));
}

ROUTES.push({ path: "welcome", title: "Set up your portfolio", section: "Home", render: pageWelcome });
