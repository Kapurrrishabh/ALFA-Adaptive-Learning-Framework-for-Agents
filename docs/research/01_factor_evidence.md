# Factor evidence for NSE equities (research, 2026-09-27)

Sources fetched directly (search engines were blocked for the research agent):
IIMA four-factor library https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/ ;
NSE methodology https://www.niftyindices.com/Methodology/Method_NIFTY_Equity_Indices.pdf ;
NSE factsheets https://www.niftyindices.com/reports/index-factsheet ;
PEAD: Harshita, Singh & Yadav (2018) https://www.scirp.org/journal/paperinformation?paperid=88060 ;
costs https://zerodha.com/charges

## Academic premia (IIMA, survivorship-adjusted, gross long-short, monthly CSV)
| Factor | CAGR 1994–2025 | Vol | Max DD | CAGR 2010–25 | CAGR 2021–25 |
|---|---|---|---|---|---|
| WML momentum (t−12..t−1) | 10.6% | 24.5% | −53.5% (2000) | 14.3% | 11.0% |
| HML value | 6.5% | 20.4% | −58.8% | 5.7% | 21.0% |
| SMB size | −3.8% | 16.7% | −86.1% | 0.0% | 2.3% |

Momentum crashes: Nov-2001 −27.6%, May-2009 −25.0%, Dec-2008 −20.3%; full-year 2009 −30.5%.

## NSE index methods (codeable)
- Nifty200 Momentum 30: R12 = P(M−1)/P(M−13)−1, R6 = P(M−1)/P(M−7)−1, σ = 1y annualised std of daily log returns;
  MR = R/σ; Z12, Z6 cross-sectional z; Z = 0.5 Z12 + 0.5 Z6; score = 1+Z (Z≥0) else 1/(1−Z); top 30;
  weight ∝ ffmcap × score, cap min(5%, 5× mcap weight); rebalance Jun/Dec; buffer: top 15 always in, holdings out beyond 45.
- Midcap150 Momentum 50: same score; excludes non-F&O names hitting circuits on ≥20% of days (6m); buffer 75 / top 25 in.
- Nifty100 Low Vol 30: 30 lowest 1y σ; weight ∝ 1/σ; quarterly; keep within 60.
- Nifty200 Quality 30: Z = ⅓Z_ROE − ⅓Z_D/E − ⅓Z_EPSvar (financials ½ROE − ½EPSvar); exclude any negative EPS in 6y; top 30; √ffmcap × score.
- Nifty200 Value 30: Z = ¼(Z_E/P + Z_B/P + Z_S/P + Z_DY).
- Nifty Alpha 50: 1y Jensen alpha vs Nifty 50, top 50 positive alpha, quarterly.
- Alpha Low-Vol 30: 50% alpha percentile + 50% low-vol percentile.

## Factsheets (31 Aug 2026, price return; most history back-filled before launch)
| Index | 5y CAGR | Since-inception CAGR | Vol | Beta |
|---|---|---|---|---|
| 200 Momentum 30 | 10.05 | 17.44 | 22.33 | 0.95 |
| Midcap150 Momentum 50 | 17.37 | 21.53 | 21.13 | 0.82 |
| 100 Low Vol 30 | 8.95 | 15.06 | 16.83 | 0.76 |
| 200 Quality 30 | 7.10 | 15.16 | 17.89 | 0.79 |
| Alpha 50 | 13.75 | 19.56 | 24.76 | 0.97 |
| Nifty 200 | 9.26 | 12.36 | 20.84 | 0.98 |

## PEAD in India
SUE = (EPS_q − EPS_{q−4}) / price (6–12 days pre-announcement); top minus bottom decile +4.8% over days +2..+64 (Nifty 500, 2002–2017), mostly from the long side (+3.4%); weaker for large caps after 2008.

## Retail recipe
Nifty 200 F&O names, NSE momentum score on month-end prices, 10–20 stocks equal weight, semi-annual rebalance, buffer (buy top N, sell beyond 1.5–2N), optional 50/50 low-vol sleeve. Round-trip ~22 bps + ₹15.34 DP charge per sell + 5–20 bps spread; keep positions ≥ ₹20k.

## Not verified
NSE index max drawdowns; Indian short-term reversal; peer-reviewed low-vol/quality/52w-high in India; EPS variability formula; current tax rates.
