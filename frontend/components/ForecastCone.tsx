"use client";

import type { Band, Outlook } from "@/utils/outlook";
import { chosenClass } from "@/utils/outlook";

const W = 720;
const H = 250;
const PAD = { top: 14, right: 58, bottom: 22, left: 8 };
// The history is 120 sessions and the forecast 5, so drawing them to one time scale would leave the
// forecast four pixels wide. The split is a reading choice and the divider line says where it is.
const SPLIT = 0.66;

const TINTS = ["#34d399", "#60a5fa", "#fb7185"];

export default function ForecastCone({ drawn }: { drawn: Outlook }) {
  const { closes, dates, bands, horizon, probabilities } = drawn;
  const picked = chosenClass(probabilities);
  const last = closes[closes.length - 1];

  const lows = bands.map((band) => band.low);
  const highs = bands.map((band) => band.high);
  const low = Math.min(...closes, ...lows);
  const high = Math.max(...closes, ...highs);
  const pad = (high - low) * 0.06 || 1;

  const plotLeft = PAD.left;
  const plotRight = W - PAD.right;
  const splitX = plotLeft + (plotRight - plotLeft) * SPLIT;
  const y = (value: number) =>
    PAD.top + (H - PAD.top - PAD.bottom) * (1 - (value - low + pad) / (high - low + 2 * pad));
  const x = (index: number) =>
    plotLeft + ((splitX - plotLeft) * index) / Math.max(1, closes.length - 1);

  const history = closes.map((close, index) => `${x(index)},${y(close)}`).join(" ");
  const yLast = y(last);
  const wedge = (band: Band) =>
    `${splitX},${yLast} ${plotRight},${y(band.high)} ${plotRight},${y(band.low)}`;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img"
           aria-label={`${drawn.symbol} closes and the ${horizon} session risk bands`}>
        <line x1={plotLeft} x2={plotRight} y1={yLast} y2={yLast} stroke="#ffffff14" strokeDasharray="2 4" />

        {/* Widest first: the classes nest inside each other, so painting narrow-first hides the calm band
            under the turbulent one entirely. */}
        {bands.map((band, index) => ({ band, index })).reverse().map(({ band, index }) => (
          <g key={band.name} opacity={picked < 0 || index === picked ? 1 : 0.28}>
            <polygon points={wedge(band)} fill={TINTS[index]}
                     fillOpacity={index === picked ? 0.16 : 0.05} />
            {/* The top class has no upper edge, so its boundary is dashed and the lines carry on past it
                rather than closing the band at a price no measurement supports. */}
            <polyline points={wedge(band)} fill="none" stroke={TINTS[index]} strokeWidth={1.4}
                      strokeDasharray={band.open ? "4 3" : undefined} />
            {band.open && [1, -1].map((way) => (
              <line key={way} x1={plotRight} x2={plotRight}
                    y1={y(way > 0 ? band.high : band.low)}
                    y2={way > 0 ? PAD.top : H - PAD.bottom}
                    stroke={TINTS[index]} strokeWidth={1.4} strokeDasharray="3 3" />
            ))}
          </g>
        ))}

        <polyline points={history} fill="none" stroke="#e5e7eb" strokeWidth={1.6} />
        <circle cx={splitX} cy={yLast} r={3} fill="#e5e7eb" />
        <line x1={splitX} x2={splitX} y1={PAD.top} y2={H - PAD.bottom} stroke="#ffffff1f" />

        <text x={plotRight + 5} y={yLast + 3.5} fill="#9ca3af" fontSize={10} fontFamily="monospace">
          {last.toFixed(2)}
        </text>
        {picked >= 0 && [bands[picked].high, bands[picked].low].map((edge, index) => (
          <text key={edge} x={plotRight + 5} y={y(edge) + (index === 0 ? -1 : 8)} fontSize={10}
                fill={TINTS[picked]} fontFamily="monospace">
            {bands[picked].open ? "≥" : ""}{edge.toFixed(2)}
          </text>
        ))}
        <text x={plotLeft} y={H - 7} fill="#6b7280" fontSize={10} fontFamily="monospace">
          {dates[0]}
        </text>
        <text x={splitX} y={H - 7} fill="#6b7280" fontSize={10} fontFamily="monospace" textAnchor="middle">
          {drawn.as_of}
        </text>
        <text x={plotRight} y={H - 7} fill="#6b7280" fontSize={10} fontFamily="monospace" textAnchor="end">
          +{horizon} sessions
        </text>
      </svg>

      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1">
        {bands.map((band, index) => (
          <span key={band.name} className="flex items-center gap-1.5 text-[11px]"
                style={{ color: index === picked ? TINTS[index] : "#6b7280" }}>
            <span className="inline-block w-2.5 h-2.5 rounded-sm"
                  style={{ background: TINTS[index], opacity: index === picked ? 1 : 0.35 }} />
            {band.name} {band.open ? "≥" : "±"}{(band.sigma * 100).toFixed(1)}%
            {probabilities && ` · ${(probabilities[index] * 100).toFixed(0)}%`}
          </span>
        ))}
      </div>
    </div>
  );
}
