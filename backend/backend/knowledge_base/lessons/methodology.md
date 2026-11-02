# Research methodology

## Look-ahead bias
Using information in a test that was not yet available at the time. For example,
using a day's close to trade at that same close. Or using restated (revised)
financials as if they were known on the original date. This system decides at
the close and fills (trades) at the next open. It computes every feature from
bars up to the decision date only. And it replays engines on truncated
(cut-short) histories.

## Survivorship bias
Testing only on companies that still exist today overstates returns. Failed and
delisted companies are missing. Any universe (the stocks tested) built from
today's index members suffers from this.

## Data snooping and multiple testing
Testing many strategies or parameters on the same data means some look good by
chance. The best of N backtests is biased upward. Remedies: fix parameters
before testing, and test on untouched data. Require significance (results
unlikely to be luck). Check that results hold across stocks and sub-periods.

## Walk-forward validation, purging and embargo
Models are trained on the past and tested on the next period, then the window
rolls forward. Labels (what a model predicts) can overlap in time. A 5-day
return label on day t overlaps the label on day t+1. So training rows whose
label window reaches into the test period must be removed ("purged"), as
described by López de Prado (2018). Random train/test splits of time series leak
future information and are never used here.

## Base rates, Brier score and calibration
A probability forecast must beat the base rate. Say a stock rose over 5 days 54%
of the time historically. Then always predicting 0.54 is the benchmark. The
Brier score is the mean squared error of probability forecasts. Brier skill
score is 1 − Brier(model) / Brier(base rate). So it is positive only when the
model adds information. A calibrated forecaster's "60%" events happen about 60%
of the time.

## Why the system does not predict exact prices
Daily stock returns are close to unpredictable. Most of their variance is noise.
Probabilities, ranges and volatility are more honest targets. Volatility is far
more predictable than direction. That is because it clusters.

## Variance ratio test
The Lo-MacKinlay variance ratio compares q-day return variance with q times
1-day return variance. Above 1 means a trend: moves tend to continue (positive
autocorrelation). Below 1 means mean reversion: moves tend to reverse. A
z-statistic tests if the gap from 1 is significant.

## Regime detection with a hidden Markov model
A two-state Gaussian (bell-curve) hidden Markov model sees returns as coming
from a calm or a turbulent state. It estimates the probability of each. The
system reports the filtered probability. That uses only data up to the current
date.

## How the decision labels work
BUY, WATCH, HOLD, SELL and AVOID are decision-support classifications (labels).
They are computed from a weighted vote across areas of analysis (domains), with
explicit gates (checks). BUY or SELL needs a strong combined score, enough
confidence, no major conflict between domains, and fundamentals or news that
agree with the combined view, not only price signals. SELL applies to a stock
you hold. The same negative evidence on a stock you do not hold reads AVOID.
WATCH means the evidence leans positive, but not strongly or cleanly enough for
BUY. A SELL that fails the same checks reads HOLD. The labels support a
decision. They are not orders and not guarantees.

## Facts, model outputs and interpretations
A fact is observed directly or worked out from reported data, such as RSI = 68.2
or revenue growth of 12%. A model output is a prediction or classification from
a model, such as a 0.57 probability of a rise, or a sentiment score. An
interpretation combines these: "momentum is strong but the stock is extended".
The system labels each piece of evidence as FACT or MODEL OUTPUT.
