# Chart patterns

## Head and shoulders
Three peaks where the middle one (the head) is clearly higher than the two
shoulders, which are at similar levels. A line through the two troughs is the
neckline. The pattern is only complete when price closes below the neckline;
before that it is "forming". The traditional target is the neckline minus the
head's height above it. The inverse head and shoulders is the bullish mirror at a
bottom.

## Double top and double bottom
Two peaks at roughly the same level separated by a meaningful trough (double
top), or two troughs separated by a peak (double bottom). Completion requires a
close through the intervening trough or peak (the neckline). Triple tops and
bottoms have three touches.

## Triangles
Converging trendlines through swing highs and lows. An ascending triangle has
flat highs and rising lows (often read as bullish); a descending triangle has
falling highs and flat lows (often bearish); a symmetric triangle has both lines
converging and no directional bias until price breaks out.

## Flags and pennants
A sharp move (the pole) followed by a short, tight consolidation. A flag's
consolidation slopes against the pole in a parallel channel; a pennant's
converges. A break in the direction of the pole completes the continuation
pattern.

## Wedges, channels and rectangles
A channel has parallel boundaries sloping up or down. A wedge has both
boundaries sloping the same way but converging; a rising wedge is often read as
bearish and a falling wedge as bullish. A rectangle has flat boundaries and marks
a trading range.

## Rounding bottom and top
A gradual U-shaped (bottom) or inverted-U (top) price path over weeks or months,
showing a slow change in the balance of buyers and sellers.

## Breakouts and false breakouts
A breakout is a close beyond a pattern boundary or recent range. A false breakout
is a breakout that reverses back inside within a few sessions; it can trap
traders and is itself sometimes a signal in the opposite direction.

## Do chart patterns predict returns?
Lo, Mamaysky and Wang (2000, Journal of Finance) used kernel regression to detect
patterns objectively and found that several conditioned return distributions,
although the practical trading value was modest. Pattern detectors also fire
often on purely random prices, which shows how easy it is to see structure in
noise. This system therefore reports, for each current pattern, how many times it
completed in the stock's own history and whether returns afterwards differed
significantly from the stock's normal behaviour.

## Numerical versus image-based detection
Patterns can be found directly from price series (swing points and fitted lines,
as here) or by running a vision model on chart images. Numerical detection is
exact, fast, and auditable. Vision models can match a human chartist's judgement
on fuzzy shapes but need labelled images and can learn artefacts of chart
rendering. Either approach must prove predictive value on out-of-sample data.
