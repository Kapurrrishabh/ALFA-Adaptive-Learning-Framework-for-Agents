"use strict";
/* The mutual fund page, #/fund/<AMFI scheme code>. Loaded after app.js, whose helpers it uses. */

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
    h("div", { class: "muted", style: "font-size:var(--fs-xs);margin-top:8px" }, `Source: ${f.source.nav}`));
  const plans = h("div", { class: "card" }, h("h3", {}, "Plans of this fund"),
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

ROUTES.push({ path: "fund", title: "Mutual fund", section: "Markets", needsArg: true, render: pageFund });
