# Market-timing and risk overlays (research, 2026-09-27)

Full text read: Faber (SSRN 962461), Moreira–Muir (NBER w22208), Harvey et al. 2018 (JPM), Moskowitz–Ooi–Pedersen 2012, Daniel–Moskowitz 2016.
Abstracts only: Kaminski–Lo 2014, Han–Zhou–Zhu (SSRN 2407199), Antonacci (SSRN 2042750), Barroso–Santa-Clara 2015.

| Overlay | Rule | Reported effect | Status |
|---|---|---|---|
| Trend filter (Faber) | hold if month-end price > 10-month SMA, else cash | S&P 1901–2012: CAGR 10.18 vs 9.32%, worst DD 42 vs 84%; invested ~70% | verified (US); Nifty untested |
| Time-series momentum | long if 12-month excess return > 0 | MOP: profits in big up/down moves, loses at sharp reversals | verified (futures) |
| Vol targeting (Harvey) | w = σ_target / σ̂, EWMA of squared returns, half-life 20d | US equity Sharpe 0.40 → 0.48–0.51, smaller left tail | verified |
| Vol-managed (Moreira–Muir) | w = c / σ̂²(last month); cap 1–1.5 | market alpha 4.9%/yr; Sharpe +25% | verified |
| Momentum crash guard (Daniel–Moskowitz) | cut momentum when market 24m return < 0 and vol high | 14 of 15 worst momentum months had bear flag = 1 | verified |
| Momentum vol scaling (Barroso–Santa-Clara) | w = 12% / σ̂(126d) of the momentum sleeve | Sharpe ~doubles, worst month −79% → −28% | numbers unverified |
| Stock stop in momentum (Han–Zhou–Zhu) | 10% stop per stock | EW worst month −49.8% → −11.4% | abstract only |
| Stops in general (Kaminski–Lo) | help only with momentum/regime switching; hurt under random walk | — | abstract only |

Tax (ClearTax / incometax.gov.in): STCG 20%, LTCG 12.5% above ₹1.25 lakh/FY, long-term > 12 months.
Rule: avoid profitable exits within ~60 days of the 12-month mark; realise short-term losses; harvest ₹1.25 lakh LTCG per FY.
India VIX / FPI flows: same-day relationships, no verified drawdown prediction → use as volatility inputs only.
Suggested overlay: exposure = trend × min(1, 15% / σ̂_Nifty); halve momentum when Nifty 24m < 0 and vol top quintile.
