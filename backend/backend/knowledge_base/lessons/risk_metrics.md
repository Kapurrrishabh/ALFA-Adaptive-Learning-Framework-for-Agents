# Risk and performance metrics

## Volatility
The annualized standard deviation of daily returns (daily standard deviation ×
√252). It measures how widely returns swing, in both directions. Volatility
clusters: calm and turbulent periods tend to persist.

## Sharpe ratio
Average excess return over the risk-free rate divided by the standard deviation
of returns, annualized. A Sharpe ratio of 1 means one unit of return per unit of
volatility. It penalises upside and downside swings equally and is estimated with
large error over short periods.

## Sortino ratio
Like Sharpe, but divides by downside deviation (volatility of returns below a
threshold), so upside volatility is not penalised.

## Stop-loss
A price set in advance at which you sell, to cap the loss on one position. It
turns an open-ended loss into a known one, and takes the decision out of the
moment. The cost is being sold out of a holding that then recovers, which gets
more likely the tighter the level. A stop does not cap the loss if the price
gaps straight past it, as it can on bad news or at the open.

## Maximum drawdown and Calmar ratio
Maximum drawdown is the largest peak-to-trough fall in value over a period. The
Calmar ratio is annualized return divided by the absolute maximum drawdown.
Drawdown is often the risk measure that matters most to an investor's ability to
stay invested.

## Beta and alpha
Beta is the covariance of a stock's returns with the benchmark divided by the
benchmark's variance: a beta of 1.3 means the stock has tended to move 1.3% for
each 1% benchmark move. Jensen's alpha is the average return left after
accounting for beta exposure and the risk-free rate.

## Value at Risk (VaR) and CVaR
Historical 95% one-day VaR is the loss exceeded on only 5% of past days. CVaR
(expected shortfall) is the average loss on those worst 5% of days, which says
more about tail risk than VaR does. Both assume the past is a guide to the
tails, which fails in regime changes.

## Correlation and diversification
Correlation measures how closely two return series move together (−1 to +1).
Combining assets with low correlation lowers portfolio volatility below the
weighted average of individual volatilities. The diversification ratio (weighted
average volatility / portfolio volatility) is above 1 when diversification works.
Correlations tend to rise in market stress, exactly when diversification is most
needed.

## Risk contribution
A holding's contribution to portfolio volatility is its weight × its marginal
contribution (covariance with the portfolio / portfolio volatility). Contributions
sum to total portfolio volatility, so a holding can supply far more risk than its
weight suggests if it is volatile or highly correlated with the rest.

## Concentration (HHI)
The Herfindahl-Hirschman Index is the sum of squared portfolio weights. Its
inverse is the "effective number of holdings": a 10-stock portfolio with one 50%
position behaves like far fewer than 10 independent bets.
