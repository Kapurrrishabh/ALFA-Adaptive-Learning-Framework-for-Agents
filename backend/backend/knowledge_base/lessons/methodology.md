# Research methodology

## Look-ahead bias
Using information in a test that would not have been available at the time, for
example using a day's close to trade at that same close, or using restated
financials as if they were known on the original date. This system decides at
the close and fills at the next open, computes every feature from bars up to the
decision date only, and replays engines on truncated histories.

## Survivorship bias
Testing only on companies that still exist today overstates returns, because
failed and delisted companies are missing. Any universe built from today's index
members suffers from this; results from such a universe are optimistic.

## Data snooping and multiple testing
Testing many strategies or parameters on the same data guarantees that some look
good by chance. The best of N backtests is biased upward. Remedies: fix
parameters before testing, test on untouched data, require significance, and
check that results hold across stocks and sub-periods.

## Walk-forward validation, purging and embargo
Models are trained on the past and tested on the following period, then the
window rolls forward. When labels overlap in time (a 5-day return label on day t
overlaps the label on day t+1), training rows whose label window reaches into the
test period must be removed ("purged"), as described by López de Prado (2018).
Random train/test splits of time series leak future information and are never
used here.

## Base rates, Brier score and calibration
A probability forecast must beat the base rate: if a stock rose over 5 days 54%
of the time historically, always predicting 0.54 is the benchmark. The Brier
score is the mean squared error of probability forecasts; Brier skill score is
1 − Brier(model) / Brier(base rate), so it is positive only when the model adds
information. A calibrated forecaster's "60%" events happen about 60% of the
time.

## Why the system does not predict exact prices
Daily stock returns are close to unpredictable; most of their variance is noise.
Probabilities, ranges, and volatility are more honest targets. Volatility is far
more forecastable than direction because it clusters.

## Variance ratio test
The Lo-MacKinlay variance ratio compares the variance of q-day returns to q times
the variance of 1-day returns. A ratio above 1 indicates trending (positive
autocorrelation), below 1 mean reversion; a z-statistic tests whether the
difference from 1 is significant.

## Regime detection with a hidden Markov model
A two-state Gaussian hidden Markov model treats returns as coming from a calm or a
turbulent state and estimates the probability of each. The system reports the
filtered probability, which uses only data up to the current date.

## How the decision labels work
BUY, WATCH, HOLD, SELL and AVOID are decision-support classifications computed
from a weighted vote of analytical domains, with explicit gates. BUY or SELL
requires a strong combined score, sufficient confidence, no major conflict
between domains, and fundamentals or news that agree with the combined view,
not only price signals. SELL applies to a stock you hold; the
same negative evidence on a stock you do not hold reads AVOID. WATCH means the
evidence leans positive but not strongly or cleanly enough for BUY; a SELL that
fails the same checks reads HOLD. The labels support a decision; they are not orders
and not guarantees.

## Facts, model outputs and interpretations
A fact is directly observed or calculated from reported data (RSI = 68.2, revenue
growth 12%). A model output is a prediction or classification from a model
(probability of a rise = 0.57, a sentiment score). An interpretation combines
several of these (momentum is strong but the stock is extended). The system
labels each piece of evidence as FACT or MODEL OUTPUT.
