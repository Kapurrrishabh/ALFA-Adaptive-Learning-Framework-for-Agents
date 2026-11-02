# Chart patterns

## Head and shoulders
Three peaks where the middle one (the head) is clearly higher than the two
shoulders, which are at similar levels. A line through the two troughs (dips) is
the neckline. The pattern is only complete when price closes below the neckline;
before that it is "forming". The traditional target is the neckline minus the
head's height above it. The inverse head and shoulders is the bullish (upward)
mirror at a bottom.

## Double top and double bottom
A double top is two peaks at about the same level, with a meaningful trough
(dip) between. A double bottom is two troughs with a peak between. Completion
needs a close through the middle trough or peak (the neckline). Triple tops and
bottoms have three touches.

## Triangles
Converging trendlines through swing highs and lows. An ascending triangle has
flat highs and rising lows, often read as bullish (up). A descending triangle
has falling highs and flat lows, often bearish (down). A symmetric triangle has
both lines closing in and no directional bias until price breaks out.

## Flags and pennants
A sharp move (the pole), then a short, tight pause (consolidation). A flag's
pause slopes against the pole in a parallel channel. A pennant's pause narrows.
A break in the pole's direction completes this continuation pattern.

## Wedges, channels and rectangles
A channel has parallel boundaries sloping up or down. A wedge has both
boundaries sloping the same way but coming together. A rising wedge is often
read as bearish (down) and a falling wedge as bullish (up). A rectangle has flat
boundaries and marks a trading range.

## Rounding bottom and top
A gradual U-shaped (bottom) or upside-down U-shaped (top) price path over weeks
or months. It marks a slow change in the balance of buyers and sellers.

## Breakouts and false breakouts
A breakout is a close beyond the boundary (edge) of a pattern or a recent range.
A false breakout reverses back inside within a few sessions. It can trap
traders. It is sometimes itself a signal in the opposite direction.

## Do chart patterns predict returns?
Lo, Mamaysky and Wang (2000, Journal of Finance) found patterns automatically
with kernel regression (a smoothing method). For several patterns, the
distribution of later returns shifted. But the trading value was modest. Pattern
detectors fire often on purely random prices, too. So it is easy to see
structure in noise. That is why this system reports how many times each current
pattern completed in the stock's own history, and whether later returns differed
from the stock's normal behaviour by more than chance.

## Numerical versus image-based detection
Patterns can be found directly from price series (swing points and fitted lines,
as here), or by a vision model reading chart images. Numerical detection is
exact, fast and easy to audit. Vision models can match a human chartist's
judgement on fuzzy shapes. But they need labelled images and can learn quirks of
how charts are drawn. Either approach must prove predictive value on new
(out-of-sample) data.
