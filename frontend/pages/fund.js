"use strict";
/* Mutual funds: the fund page (#/fund/<AMFI scheme code>), fund rows for search, and the funds card on the
   portfolio page. Loaded after app.js, whose helpers it uses. */

const fundHref = (code) => `#/fund/${encodeURIComponent(code)}`;
const planName = (p) => [p.plan, p.option].filter(Boolean).join(" · ") || "AMFI names no plan";
// a search hit is a fund with its plans, best first, so the row opens the first plan
const fundRow = (f) => h("a", { href: fundHref(f.plans[0].code) },
  h("div", { style: "min-width:0" }, h("div", { class: "t" }, f.fund), h("div", { class: "s" }, `${f.amc} · ${f.category}`)),
  h("div", { class: "right muted" }, `${f.plans.length} ${f.plans.length === 1 ? "plan" : "plans"}`));

function addFundModal(ctx, sc) {
  const f = { side: h("select", {}, h("option", {}, "BUY"), h("option", {}, "SELL")), units: h("input", { type: "number", min: "0", step: "any" }),
    nav: h("input", { type: "number", min: "0", step: "any", value: String(sc.nav) }),
    trade_date: h("input", { type: "date", value: new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Kolkata" }).format(new Date()) }),
    fees: h("input", { type: "number", min: "0", step: "0.01", value: "0" }) };
  for (const [name, input] of Object.entries(f)) input.name = name;
  const save = async () => {
    await api("/portfolio/funds/transactions", { method: "POST", body: JSON.stringify({ scheme_code: sc.code, side: f.side.value, units: +f.units.value, nav: +f.nav.value, trade_date: f.trade_date.value, fees: +f.fees.value }) });
    toast("Saved to portfolio");
  };
  Dialog(ctx, `Add ${sc.fund}`, [h("p", { class: "muted" }, planName(sc)),
    h("div", { class: "grid", style: "grid-template-columns:1fr 1fr" }, Field("Type", f.side), Field("Date", f.trade_date)),
    h("div", { class: "grid", style: "grid-template-columns:1fr 1fr" }, Field("Units", f.units), Field("NAV per unit (₹)", f.nav, `Latest: ${inr(sc.nav, 4)} on ${sc.nav_date}`)),
    Field("Fees (₹)", f.fees, "Stamp duty and other charges for the whole order. Leave 0 if you paid none.")], ["Save", save]);
  f.units.focus();
}

async function pageFund(view, ctx) {
  view.append(h("div", { class: "card hero" }, skel(26, "40%"), skel(34, "25%")), h("div", { class: "card", style: "margin-top:18px" }, skel(260)));
  const f = await api(`/ui/funds/${encodeURIComponent(ctx.arg)}`, { signal: ctx.signal }), sc = f.scheme;
  const head = h("div", { class: "card hero" },
    h("div", { class: "spread stock-head" }, h("div", {}, h("h2", {}, sc.fund), h("div", { class: "muted" }, `${sc.amc} · ${sc.category}`), h("div", { class: "muted" }, planName(sc))),
      h("button", { class: "btn", onclick: () => addFundModal(ctx, sc) }, "+ Add to portfolio")),
    h("div", { class: "row", style: "margin-top:14px;align-items:baseline" }, h("div", { class: "price num" }, inr(sc.nav, 4)), h("div", { class: "muted" }, `NAV on ${sc.nav_date}`)),
    h("div", { class: "muted", style: "font-size:12px;margin-top:8px" }, `Source: ${f.source.nav}`));
  const plans = h("div", { class: "card" }, h("h3", {}, "Plans of this fund"),
    h("p", { class: "sub" }, "Direct plans cost less. Regular plans pay a distributor from the fund's costs, so they grow more slowly."),
    h("div", { class: "list" }, ...f.plans.map((p) => h("a", { href: fundHref(p.code), "aria-current": p.code === sc.code ? "page" : null },
      h("div", {}, h("div", { class: "t" }, planName(p)), h("div", { class: "s" }, p.is_ambiguous ? `Code ${p.code} · AMFI names another plan the same way` : `Code ${p.code}`)),
      h("div", { class: "right num" }, inr(p.nav, 4))))));
  const record = h("div", { class: "card" }, h("h3", {}, "NAV over time"));
  if (f.history) {
    const a = f.analytics, host = h("div", { class: "chart" });
    const chip = (k, v, c = "") => h("div", { class: "stat-chip" }, h("div", { class: "k" }, k), h("div", { class: `v num ${c}` }, v));
    record.append(h("p", { class: "sub" }, `From ${f.source.history}; it ${f.history_check}.`),
      h("div", { class: "chips-row" }, ...Object.entries(a.returns).map(([k, r]) => chip(k, r ? pct(r.pct, 1) + (r.annualised ? " a year" : "") : "—", r ? cls(r.pct) : "")),
        a.volatility_1y_pct != null ? chip("Volatility, 1 year", `${num(a.volatility_1y_pct, 1)}%`) : null), host, h("p", {}, f.summary));
    const draw = () => lineChart(host, f.history.dates, [{ name: "NAV", values: f.history.nav, color: "--series-1" }], `NAV of ${sc.fund}, ${planName(sc)}`, { fmt: (v) => inr(v), height: 260 });
    requestAnimationFrame(draw); ctx.redraw(draw);
  } else record.append(h("p", { class: "muted" }, `No NAV history to show: ${f.history_error}.`));
  const lessons = h("div", { class: "card" }, h("h3", {}, "Learn about funds"), ...f.lessons.map((l) => h("details", {}, h("summary", {}, l.title), h("p", {}, l.text))));
  view.replaceChildren(h("div", { class: "grid cols-2" }, h("div", { class: "grid" }, head, record), h("div", { class: "grid", style: "align-content:start" }, plans, lessons)),
    h("p", { class: "footer-note" }, "Decision support, not investment advice. Past NAVs do not tell you future returns."));
}

function fundsCard(ctx, held) {
  const stat = ([k, v]) => h("div", { class: "stat" }, h("div", { class: "k" }, k), h("div", { class: "v num" }, v));
  const table = h("div", { class: "table-wrap" }, h("table", {},
    h("tr", {}, h("th", {}, "Fund"), h("th", { class: "r" }, "Units"), h("th", { class: "r" }, "Avg / NAV"), h("th", { class: "r" }, "Value"), h("th", { class: "r" }, "Returns")),
    ...held.holdings.map((x) => h("tr", {}, h("td", {}, h("div", { class: "t" }, h("a", { href: fundHref(x.code) }, x.fund)), h("div", { class: "s" }, `${planName(x)} · NAV on ${x.nav_date}`)),
      h("td", { class: "r num" }, num(x.units, 3)), h("td", { class: "r num" }, h("div", {}, inr(x.avg_nav, 4)), h("div", { class: "muted" }, inr(x.nav, 4))),
      h("td", { class: "r num" }, inr(x.value, 0)), h("td", { class: `r num ${cls(x.gain)}` }, h("div", {}, inr(x.gain, 0)), h("div", {}, pct(x.gain_pct)))))));
  const txs = h("table", {}, h("tr", {}, h("th", {}, "Date"), h("th", {}, "Fund"), h("th", {}, "Type"), h("th", { class: "r" }, "Units"), h("th", { class: "r" }, "NAV"), h("th", {})),
    ...held.transactions.map((t) => h("tr", {}, h("td", {}, t.trade_date), h("td", {}, h("a", { href: fundHref(t.scheme_code) }, `Code ${t.scheme_code}`)), h("td", { class: t.side === "BUY" ? "up" : "down" }, t.side),
      h("td", { class: "r num" }, num(t.units, 3)), h("td", { class: "r num" }, inr(t.nav, 4)),
      h("td", { class: "r" }, h("button", { class: "btn small", onclick: () => Dialog(ctx, "Delete this fund transaction?", `${t.side} ${num(t.units, 3)} units of code ${t.scheme_code} at ${inr(t.nav, 4)} on ${t.trade_date}.`,
        ["Delete", async () => { await api(`/portfolio/funds/transactions/${t.id}`, { method: "DELETE" }); route(); }]) }, "Delete")))));
  return h("div", { class: "card" }, h("h3", {}, `Mutual funds (${held.holdings.length})`),
    h("div", { class: "grid cols-4" }, ...[["Invested", inr(held.invested)], ["Value", inr(held.value)], ["Returns", `${inr(held.gain)} (${pct(held.gain_pct)})`]].map(stat)),
    held.holdings.length ? table : h("p", { class: "muted" }, "You hold no fund units now."),
    h("h3", { style: "margin-top:16px" }, "Fund transactions"), txs,
    h("p", { class: "muted", style: "font-size:12px" }, `Valued at each fund's latest NAV from ${held.source}. To add a fund, search for it and press “Add to portfolio” on its page.`));
}

ROUTES.push({ path: "fund", title: "Mutual fund", section: "Markets", needsArg: true, render: pageFund });
