# Risk and performance metrics

## Volatility
Volatility measures how much returns move, up or down. It is the standard
deviation of daily returns, annualized: daily standard deviation × √252.
Volatility clusters. Calm and stormy periods tend to persist.

## Sharpe ratio
Average excess return over the risk-free rate, divided by the standard deviation
of returns. Both are annualized. A Sharpe ratio of 1 means one unit of return
per unit of volatility. It punishes upside and downside moves alike. Over short
periods, the estimate has a large error.

## Sortino ratio
Like Sharpe, but divides by downside deviation, the volatility of returns below
a threshold. So upside moves are not punished.

## Stop-loss
A price set in advance at which you sell, to cap the loss on one position. It
turns an open-ended loss into a known one, and takes the decision out of the
moment. The cost is being sold out of a holding that then recovers, which gets
more likely the tighter the level. But a stop does not cap the loss if the price
gaps straight past it. That can happen on bad news or at the open.

## Maximum drawdown and Calmar ratio
Maximum drawdown is the largest peak-to-trough fall in value over a period. The
Calmar ratio is annualized return divided by the absolute size of that fall.
Drawdown is often the risk measure that matters most to whether an investor can
stay invested.

## Beta and alpha
Beta is the covariance of a stock's returns with the benchmark, divided by the
benchmark's variance. A beta of 1.3 means the stock has tended to move 1.3% for
each 1% benchmark move. Jensen's alpha is the average return left after allowing
for beta exposure and the risk-free rate.

## Value at Risk (VaR) and CVaR
Historical 95% one-day VaR is the loss exceeded on only 5% of past days. CVaR
(expected shortfall) is the average loss on those worst 5% of days. It says more
than VaR about tail risk (rare, large losses). Both assume the past is a guide
to the tails, which fails in regime changes.

## Correlation and diversification
Correlation measures how closely two sets of returns move together (−1 to +1). A
mix of assets with low correlation moves less than its parts do on average. The
diversification ratio (weighted average volatility / portfolio volatility) is
above 1 when diversification works. In market stress, correlations tend to rise.
That is just when you need them low.

## Risk contribution
A holding's contribution to portfolio volatility is its weight × its marginal
contribution (covariance with the portfolio / portfolio volatility). These sum
to the total. So a holding can add far more risk than its weight suggests, if it
is volatile or highly correlated with the rest.

## Concentration (HHI)
The Herfindahl-Hirschman Index is the sum of squared portfolio weights. Its
inverse is the "effective number of holdings". A 10-stock portfolio with one 50%
position acts like far fewer than 10 independent bets.
