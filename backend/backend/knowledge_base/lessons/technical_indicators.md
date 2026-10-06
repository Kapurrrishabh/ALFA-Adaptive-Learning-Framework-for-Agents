# Technical indicators

## RSI (Relative Strength Index)
RSI measures the speed of recent price changes on a 0–100 scale. With Wilder's
method over 14 periods: average gain and average loss are smoothed with a factor
of 1/14, RS = average gain / average loss, and RSI = 100 − 100 / (1 + RS).
Readings above 70 are conventionally called overbought and below 30 oversold.
In a strong uptrend RSI can stay above 70 for weeks, so a high reading describes
strong momentum at least as much as it predicts a reversal. This system treats
RSI extremes as weak evidence and checks them against trend and volume.

## MACD (Moving Average Convergence Divergence)
MACD line = 12-period EMA − 26-period EMA of closing prices. The signal line is a
9-period EMA of the MACD line, and the histogram is MACD − signal. A positive
histogram means short-term momentum is above its recent average. MACD is a
lagging, smoothed momentum measure; crossovers are frequent in sideways markets
and produce many false signals there.

## Moving averages (SMA, EMA, WMA)
A simple moving average (SMA) is the arithmetic mean of the last N closes. An
exponential moving average (EMA) weights recent prices more, with smoothing
factor 2/(N+1). A weighted moving average (WMA) uses linearly increasing
weights. The 50-day and 200-day averages are widely watched: price above a
rising 200-day average is a common definition of a long-term uptrend. A 50-day
average crossing above the 200-day is called a golden cross; crossing below, a
death cross. Crossovers lag price by construction.

## Bollinger Bands
Bollinger Bands place bands two standard deviations above and below a 20-period
SMA. Band width expands with volatility. A close outside the bands says the move
was large relative to recent volatility; it is not by itself a buy or sell
signal. Very narrow bands (a "squeeze") mark low-volatility periods that often
precede larger moves in either direction.

## ATR (Average True Range)
True range is the largest of: high − low, |high − previous close|, and
|low − previous close|. ATR is its Wilder-smoothed average over 14 periods and
measures typical daily movement in price units. ATR is useful for sizing stops
and for judging whether a move is large for this particular stock.

## Stochastic oscillator
%K = 100 × (close − lowest low over N) / (highest high over N − lowest low over
N), usually with N = 14; %D is a 3-period average of %K. It shows where the
close sits within the recent range.

## Volume, OBV and accumulation/distribution
Volume confirms price moves: a breakout on volume well above its 20-day average
is more credible than one on light volume. On-Balance Volume (OBV) adds the
day's volume when the close rises and subtracts it when it falls. The
accumulation/distribution line weights volume by where the close falls within
the day's range. Divergence between price and these lines is used as a warning,
but it is weak evidence on its own.

## Support and resistance
Support is a price zone where declines have repeatedly stopped; resistance is a
zone where advances have stalled. This system finds swing highs and lows (bars
that are the extreme within five bars on each side) and clusters those within
about 1.5% into zones. More touches make a stronger zone. A decisive close
through a zone on high volume is a breakout (above resistance) or a breakdown
(below support).

## Trend structure (higher highs and higher lows)
An uptrend is a sequence of higher swing highs and higher swing lows; a
downtrend has lower highs and lower lows. A failure to make a new higher high in
an uptrend is an early sign the trend may be weakening.

## Momentum and rate of change
Rate of change (ROC) is the percentage change over N periods. Time-series
momentum—being long when the past 6–12 month return is positive—is one of the
better-documented effects in the academic literature across asset classes,
although it suffers sharp reversals ("momentum crashes") after market bottoms.
