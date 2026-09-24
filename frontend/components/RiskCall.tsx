"use client";

import { MinusCircle } from "lucide-react";

import ForecastCone from "@/components/ForecastCone";
import type { Outlook } from "@/utils/outlook";
import { chosenClass } from "@/utils/outlook";

const TINTS = ["text-emerald-400", "text-blue-400", "text-rose-400"];
const BARS = ["bg-emerald-400", "bg-blue-400", "bg-rose-400"];

/** `stated` is the confidence out of the snapshot, which is the calibrated one. Taken from there rather
 *  than recomputed from the probabilities so the panel and the answer quote the same figure. */
export default function RiskCall({ drawn, stated }: { drawn: Outlook; stated?: string }) {
  const { probabilities, bands, model, direction, horizon } = drawn;
  const picked = chosenClass(probabilities);
  const beaten = direction.accuracy >= direction.baseline;

  return (
    <section className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-5">
      <div className="flex items-baseline justify-between gap-3 mb-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-gray-400">
          The call, {horizon} sessions out
        </h2>
        <span className="text-[11px] font-mono text-gray-600">
          {model.calibrated ? "calibrated confidence" : "uncalibrated confidence"}
        </span>
      </div>

      {picked < 0 ? (
        <p className="text-sm text-gray-500">
          This instrument has fewer than {model.window + 2} bars on file, so the head was not run and the
          evidence carries no outlook for the answer to quote either.
        </p>
      ) : (
        <>
          <div className="flex items-end gap-3">
            <span className={`text-4xl font-black tracking-tight ${TINTS[picked]}`}>
              {bands[picked].name}
            </span>
            {stated && <span className="text-lg font-mono text-gray-400 pb-1">{stated}</span>}
          </div>
          <p className="text-[11px] text-gray-600 mt-1.5 leading-relaxed">
            Daily moves around {bands[picked].open ? "at least " : ""}
            {(bands[picked].sigma * 100).toFixed(1)}% either way over the window — how far it travels, not
            which way.
          </p>

          <div className="mt-4 space-y-1.5">
            {bands.map((band, index) => (
              <div key={band.name} className="flex items-center gap-2.5">
                <span className="w-20 text-[11px] text-gray-500 font-mono">{band.name}</span>
                <span className="flex-1 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                  <span className={`block h-full rounded-full ${BARS[index]}`}
                        style={{ width: `${(probabilities![index] * 100).toFixed(1)}%`,
                                 opacity: index === picked ? 1 : 0.4 }} />
                </span>
                <span className="w-10 text-[11px] text-gray-400 font-mono text-right">
                  {(probabilities![index] * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>

          <div className="mt-5">
            <ForecastCone drawn={drawn} />
          </div>
        </>
      )}

      <div className="mt-5 pt-4 border-t border-white/[0.06]">
        <div className="flex items-center gap-2">
          <MinusCircle size={15} className="text-amber-400" />
          <span className="text-sm font-bold text-amber-300">
            {beaten ? "Direction" : "No buy, sell or hold"}
          </span>
        </div>
        <p className="text-[11px] text-gray-500 mt-1.5 leading-relaxed">
          A week-ahead up-or-down call was trained and scored{" "}
          <span className="font-mono text-gray-300">{(direction.accuracy * 100).toFixed(1)}%</span> against
          a majority baseline of{" "}
          <span className="font-mono text-gray-300">{(direction.baseline * 100).toFixed(1)}%</span> — worse
          than always naming the commonest outcome. So the agent states no view, and this panel shows the
          measurement instead of a badge it cannot support.
        </p>
      </div>
    </section>
  );
}
