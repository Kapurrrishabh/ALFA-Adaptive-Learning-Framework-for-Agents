"use client";

import { useState } from "react";
import type { Scenarios } from "@/utils/scenarios";

const W = 720;
const H = 230;
const PAD = { top: 14, right: 92, bottom: 22, left: 8 };
// One hue, darker towards the middle: the bands nest, so shade says which is which without a legend.
const HUE = "#60a5fa";

export default function ScenarioFan({ drawn }: { drawn: Scenarios }) {
  const { fan, examples, close, steps } = drawn;
  const [hover, setHover] = useState<number | null>(null);

  const every = [close, ...fan["0.05"], ...fan["0.95"], ...examples.flat()];
  const low = Math.min(...every);
  const high = Math.max(...every);
  const pad = (high - low) * 0.06 || 1;
  const x = (step: number) => PAD.left + ((W - PAD.left - PAD.right) * step) / steps;
  const y = (value: number) =>
    PAD.top + (H - PAD.top - PAD.bottom) * (1 - (value - low + pad) / (high - low + 2 * pad));
  // Step 0 is today's close, so every band and path starts from the one point both sides agree on.
  const line = (values: number[]) => [close, ...values].map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const band = (lower: number[], upper: number[]) =>
    `${line(upper)} ${[close, ...lower].map((v, i) => `${x(i)},${y(v)}`).reverse().join(" ")}`;

  const pick = (event: React.MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const step = Math.round((((event.clientX - box.left) / box.width) * W - PAD.left) /
                            ((W - PAD.left - PAD.right) / steps));
    setHover(step >= 1 && step <= steps ? step : null);
  };
  const measured = drawn.model.measured;

  return (
    <div className="rounded-xl border border-white/[0.06] bg-white/[0.015] px-4 py-3">
      <div className="flex items-baseline justify-between gap-3 mb-1">
        <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">
          {drawn.paths} possible paths, next {steps} sessions
        </div>
        <div className="text-[11px] text-gray-500 font-mono">
          drawn by {drawn.model.name === "generative" ? "the return generator" : "GARCH-t"}
        </div>
      </div>
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" onMouseMove={pick}
             onMouseLeave={() => setHover(null)}
             aria-label={`${drawn.symbol}: 5th to 95th percentile of sampled prices over ${steps} sessions`}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(close)} y2={y(close)} stroke="#ffffff14"
                strokeDasharray="2 4" />
          <polygon points={band(fan["0.05"], fan["0.95"])} fill={HUE} fillOpacity={0.12} />
          <polygon points={band(fan["0.25"], fan["0.75"])} fill={HUE} fillOpacity={0.26} />
          {examples.map((path, index) => (
            <polyline key={index} points={line(path)} fill="none" stroke="#9ca3af" strokeOpacity={0.35}
                      strokeWidth={1} />
          ))}
          <polyline points={line(fan["0.5"])} fill="none" stroke={HUE} strokeWidth={2} />
          {(["0.95", "0.5", "0.05"] as const).map((q) => (
            <text key={q} x={W - PAD.right + 5} y={y(fan[q][steps - 1]) + 3.5} fontSize={10}
                  fill="#9ca3af" fontFamily="monospace">
              {fan[q][steps - 1].toFixed(2)} {q === "0.5" ? "mid" : q === "0.95" ? "p95" : "p5"}
            </text>
          ))}
          {hover !== null && (
            <line x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={H - PAD.bottom} stroke="#ffffff40" />
          )}
          <text x={PAD.left} y={H - 7} fill="#6b7280" fontSize={10} fontFamily="monospace">{drawn.as_of}</text>
          <text x={W - PAD.right} y={H - 7} fill="#6b7280" fontSize={10} fontFamily="monospace" textAnchor="end">
            +{steps} sessions
          </text>
        </svg>
        {hover !== null && (
          <div className="absolute top-1 left-2 rounded-md bg-black/80 border border-white/10 px-2 py-1 text-[11px]
                          font-mono text-gray-300 pointer-events-none">
            +{hover}: {fan["0.05"][hover - 1].toFixed(2)} – {fan["0.95"][hover - 1].toFixed(2)}, middle{" "}
            {fan["0.5"][hover - 1].toFixed(2)}
          </div>
        )}
      </div>
      <p className="text-[11px] text-gray-500 mt-1.5">
        Dark band: half of the paths. Light band: 90%. Grey lines: five single paths. {drawn.direction}
      </p>
      <p className="text-[11px] text-gray-600 mt-1">
        Measured from {drawn.model.test_from} on {measured.scored.toLocaleString()} returns: log score{" "}
        {measured.vs_garch > 0 ? "+" : ""}{measured.vs_garch.toFixed(4)} ± {measured.vs_garch_error.toFixed(4)} nats
        against GARCH-t (negative is the generator ahead).
      </p>
    </div>
  );
}
