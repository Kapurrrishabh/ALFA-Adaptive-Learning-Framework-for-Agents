# India market data sources (tested 2026-09-27)

Access: `nsearchives.nseindia.com` / `archives.nseindia.com` answer plain requests with a browser User-Agent.
`www.nseindia.com/api/*` mostly work with UA + `Referer: https://www.nseindia.com/`; homepage and BSE API need
`curl_cffi` with `impersonate="chrome"` (TLS fingerprinting).

| # | Data | Endpoint | History | Evidence of value |
|---|---|---|---|---|
| 1 | Full bhavcopy with delivery % | `nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv`; pre-2020 `archives/equities/mto/MTO_DDMMYYYY.DAT` | Dec 2019+ (MTO 2010+) | unproven — test ourselves |
| 2 | India VIX | `www.nseindia.com/api/historicalOR/vixhistory?from=DD-MM-YYYY&to=DD-MM-YYYY` | multi-year | forecasts realised vol; risk sizing |
| 2 | Participant OI (FII/DII/pro/client) | `nsearchives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv` | daily files | positioning / regime |
| 3 | Insider (PIT) trades | `api/corporates-pit?index=equities&from_date=&to_date=` (to Apr-2026, full JSON); `api/corporates-pit-gg` (May-2026+, XBRL links) | 2020+ | promoter market purchases → positive abnormal returns (best stock-level signal) |
| 4 | Pledges | `api/corporate-pledgedata-sast3132?index=equities` | events | crash-risk flag |
| 4 | Shareholding | `api/corporate-share-holdings-master?index=equities&symbol=X` | 21 quarters | promoter trend |
| 5 | FII/DII cash | `api/fiidiiTradeReact` | latest day only — store daily | follows returns; regime only |
| 6 | Announcements / board meetings | `api/corporate-announcements`, `api/event-calendar` | daily | event awareness |
| 7 | Bulk/block deals | `api/historicalOR/bulk-block-short-deals?optionType=bulk_deals&from=&to=` | history | weak after costs |
| — | Option chain | `api/option-chain-v3?type=Indices&symbol=NIFTY&expiry=DD-Mon-YYYY` | live only | PCR |
| — | Constituents | `archives.nseindia.com/content/indices/ind_nifty500list.csv` | today only | no free membership history (survivorship) |

Brokers: Kite historical needs ₹500/month; Upstox/Angel/Dhan/Fyers/Breeze free for account holders (unverified depth).
Packages on Py3.13: nselib 2.5.1 (delivery, bulk, VIX, participant OI OK), jugaad-data OK, nsepython fiidii OK.
Old endpoints now dead: `api/historical/*` (503), `option-chain-indices` (404), `corporates-pit` empty after Apr-2026.
Paper citations for predictive value were from memory (unverified).
