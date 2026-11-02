"use strict";
/* #/import: a broker file is checked by the server, the person fills every gap it shows, then it is saved in one go. */
Object.assign(ICONS, { tick: "M5 12l5 5 9-10", cross: "M6 6l12 12M18 6L6 18" });

// Where each broker keeps its export. Like the file layouts, this comes from third-party guides
const IMPORT_SOURCES = [
  ["Zerodha holdings", "On console.zerodha.com, open Portfolio, then Holdings, and download the list as XLSX."],
  ["Zerodha tradebook", "On console.zerodha.com, open Reports, then Tradebook. Pick Equity and the dates, then download it as XLSX or CSV."],
  ["Groww stocks", "In the Groww app, open Profile, then Reports, then Stocks, and download the holdings statement. This may differ in your app."],
  ["Our CSV template", "Download the template and add one row per trade: the symbol or ISIN, BUY or SELL, the quantity, the price per share, and the date written like 2026-01-15."],
];

const many = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

function importRow(r) {
  const x = { r, symbol: r.listing ? r.listing.symbol : "", date: r.trade_date || "", price: r.price == null ? "" : String(r.price) };
  x.status = () => r.state === "not_found" ? ["cross", "Left out", "It is not in NSE's lists of shares, ETFs and REITs."]
    : x.symbol === "none" ? ["cross", "Left out", "You chose none of these."]
      : !r.quantity ? ["cross", "Left out", "The file shows 0 shares you can sell. Add them by hand if you own them."]
        : !x.symbol ? ["alert", "To do", "Pick the stock."] : !x.date ? ["alert", "To do", "Enter the buy date."]
          : !(+x.price > 0) ? ["alert", "To do", "Enter the price per share."] : ["tick", "Ready", ""];
  x.useDate = (day) => { if (x.dateInput && !x.date) { x.date = x.dateInput.value = day; } };
  x.label = r.symbol || r.name || r.isin || `row ${r.line}`;
  return x;
}

async function pageImport(view, ctx) {
  let p = null;
  try { p = await api("/portfolio", { signal: ctx.signal }); } catch (e) { if (e.status !== 404) throw e; }
  if (!p) {
    view.append(EmptyState("pie", "Set up your portfolio first", "Choose your risk level, time horizon and cash, then come back to bring in a file.",
      h("a", { class: "btn primary", href: "#/welcome" }, "Set up my portfolio")));
    return;
  }
  const file = h("input", { type: "file", accept: ".xlsx,.csv,.pdf" }), password = h("input", { type: "password", autocomplete: "off" });
  const fileField = Field("Your file", file, "An Excel (.xlsx) or CSV file from your broker, or our template. Nothing is saved until you confirm.");
  const passwordBox = h("div", { hidden: true }, Field("PDF password", password, "PDF statements are locked. The password only opens this file and is never kept."));
  file.addEventListener("change", () => { passwordBox.hidden = !(file.files[0] && file.files[0].type === "application/pdf"); });
  const check = h("button", { class: "btn primary" }, "Check this file"), out = h("div", { class: "grid", style: "margin-top:18px" });
  const upload = async () => {
    if (!file.files[0]) return fileField.showError("Choose a file first.");
    fileField.showError(null); check.disabled = true;
    const body = new FormData(); body.append("file", file.files[0]);
    if (!passwordBox.hidden) body.append("password", password.value);
    out.replaceChildren(h("div", { class: "card" }, h("p", { class: "muted" }, "Reading your file…"), skel(120)));
    try { showPreview(await api("/portfolio/import/preview", { method: "POST", body })); }
    catch (e) { out.replaceChildren(e.status === 501 ? h("div", { class: "card" }, h("p", {}, e.message)) : ErrorState(e, upload)); }
    finally { check.disabled = false; }
  };
  check.addEventListener("click", upload);
  view.append(h("div", { class: "grid cols-4 import-sources" }, ...IMPORT_SOURCES.map(([name, how], i) => h("div", { class: "card" }, h("h3", {}, name),
      h("details", {}, h("summary", {}, "How do I download this?"), h("p", {}, how)),
      i === IMPORT_SOURCES.length - 1 ? h("a", { class: "btn small", href: "/static/import-template.csv", download: "stockintel-import.csv" }, "Download the template") : null))),
    h("div", { class: "card", style: "margin-top:18px" }, fileField, passwordBox, h("div", { class: "row", style: "margin-top:12px" }, check)), out);

  const showPreview = (pv) => {
    const rows = pv.rows.map(importRow), c = pv.counts;
    let mode = "add";
    const go = h("button", { class: "btn primary" }), why = h("p", { class: "muted", role: "status" });
    const update = () => {
      for (const x of rows) { const [ic, word, reason] = x.status(); x.state.replaceChildren(h("span", { class: `pill ${{ tick: "green", alert: "amber", cross: "red" }[ic]}` }, icon(ic), word), h("div", { class: "muted" }, reason)); }
      const ready = rows.filter((x) => x.status()[0] === "tick"), todo = rows.filter((x) => x.status()[0] === "alert").length;
      go.textContent = `${mode === "add" ? "Add" : "Replace with"} ${many(ready.length, "transaction")}`;
      go.disabled = todo > 0 || !ready.length;
      why.textContent = todo ? `${todo} ${todo === 1 ? "row still needs" : "rows still need"} something. Each row says what.` : ready.length ? "" : "No row in this file can be added.";
      return ready;
    };
    const cell = (x) => {
      if (x.r.state === "matched") return h("div", {}, h("b", {}, x.symbol), x.r.listing.name ? h("div", { class: "muted" }, x.r.listing.name) : null);
      if (x.r.state === "not_found") return "—";
      const pick = h("select", { "aria-label": `Stock for ${x.label}` }, h("option", { value: "" }, "Pick the stock"),
        ...x.r.choices.map((ch) => h("option", { value: ch.symbol }, ch.name ? `${ch.symbol} · ${ch.name}` : ch.symbol)), h("option", { value: "none" }, "None of these"));
      pick.addEventListener("change", () => { x.symbol = pick.value; update(); });
      return pick;
    };
    const input = (x, key, attrs) => {
      const el = h("input", attrs); el.addEventListener("input", () => { x[key] = el.value; update(); });
      return el;
    };
    const table = h("table", {}, h("tr", {}, h("th", {}, "In your file"), h("th", {}, "Stock"), h("th", {}, "Type"), h("th", { class: "r" }, "Qty"), h("th", {}, "Buy date"), h("th", {}, "Price (₹)"), h("th", {}, "Status")),
      ...rows.map((x) => {
        x.state = h("td");
        if (!x.r.trade_date) x.dateInput = input(x, "date", { type: "date", "aria-label": `Date for ${x.label}` });
        const held = pv.held[x.symbol];
        return h("tr", {}, h("td", {}, h("div", {}, x.label), h("div", { class: "muted" }, [x.r.isin, x.r.line && `line ${x.r.line}`].filter(Boolean).join(" · ")),
          x.r.note || x.r.doubt ? h("div", { class: "muted", style: "font-size:12px" }, [x.r.doubt, x.r.note].filter(Boolean).join(" ")) : null),
          h("td", {}, cell(x), held > 0 ? h("div", { class: "muted" }, `You hold ${num(held, 0)} now`) : null), h("td", {}, x.r.side), h("td", { class: "r num" }, num(x.r.quantity, 0)),
          h("td", {}, x.r.trade_date || x.dateInput), h("td", {}, x.r.price != null ? inr(x.r.price) : input(x, "price", { type: "number", min: "0", step: "0.05", "aria-label": `Price per share for ${x.label}` })), x.state);
      }));
    const undated = rows.some((x) => x.dateInput);
    const bulk = pv.as_of && undated ? h("div", { class: "note", style: "margin-top:12px" }, h("button", { class: "btn small", onclick: () => { for (const x of rows) x.useDate(pv.as_of); update(); } }, `Use the statement date (${pv.as_of}) for all`),
      " Tax on a sale depends on how long you held the shares, so enter the real buy dates where you know them.") : null;
    const save = async (ready) => {
      await api("/portfolio/import/confirm", { method: "POST", body: JSON.stringify({ mode, stocks: ready.map((x) => ({ symbol: x.symbol, side: x.r.side, quantity: x.r.quantity, price: +x.price, trade_date: x.date, ...(x.r.fees != null ? { fees: x.r.fees } : {}) })) }) });
      toast(`Saved ${many(ready.length, "transaction")}`); location.hash = "#/portfolio";
    };
    go.addEventListener("click", async () => {
      const ready = update();
      if (mode === "replace") return Dialog(ctx, "Replace your transactions?", `This deletes your ${many(p.transactions.length, "transaction")} and saves ${many(ready.length, "transaction")} from this file instead.`, ["Replace", () => save(ready)]);
      go.disabled = true;
      try { await save(ready); } catch (e) { why.textContent = `Saving failed: ${errorWords(e)}`; go.disabled = false; }
    });
    const modes = p.transactions.length ? h("div", { class: "ctl" }, h("span", { class: "k" }, "What to do with the transactions you already have"),
      Segmented([["add", "Keep them and add these"], ["replace", "Replace them with these"]], mode, (v) => { mode = v; update(); })) : null;
    out.replaceChildren(h("div", { class: "card" }, h("h3", {}, `${pv.source}${pv.as_of ? `, as on ${pv.as_of}` : ""}`),
      h("p", {}, `${many(rows.length, "row")}: ${c.matched} found, ${c.check} to check, ${c.not_found} not found.`),
      pv.warnings.length ? h("ul", { class: "checks con" }, ...pv.warnings.map((w) => h("li", {}, w))) : null,
      h("div", { class: "table-wrap" }, table), bulk, h("div", { class: "grid", style: "margin-top:16px" }, modes, h("div", { class: "row" }, go), why)));
    update();
  };
}

ROUTES.push({ path: "import", title: "Bring in your portfolio", section: "Portfolio", label: "Import", render: pageImport });
