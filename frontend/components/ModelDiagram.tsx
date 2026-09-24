"use client";

import { ArrowDown } from "lucide-react";

import type { ModelCard } from "@/utils/outlook";

/** The blocks a window passes through, per tower. Read off `serving`, so the picture is of whatever is
 *  actually answering rather than of whichever tower the code was written around. */
function stages(model: ModelCard): { title: string; detail: string }[] {
  const window = `${model.window} bars x ${model.channels} channels`;
  if (model.serving === "persistence") {
    return [
      { title: "Trailing window", detail: `${model.window} bars of closes` },
      { title: "Daily log returns", detail: "standard deviation over the window" },
      { title: "Bucket", detail: "which tertile that value falls in" },
      { title: "Lookup", detail: "how often that bucket was right in training" },
    ];
  }
  if (model.serving === "transformer") {
    return [
      { title: "Price window", detail: window },
      { title: "Patch embedding", detail: `overlapping patches to width ${model.dim}` },
      { title: "Transformer encoder", detail: "self-attention over the patches" },
      { title: "Mean over patches", detail: `one vector of ${model.dim}` },
      { title: "Linear", detail: `${model.dim} to ${model.classes.length} classes` },
    ];
  }
  return [
    { title: "Price window", detail: window },
    { title: "GRU", detail: `one pass, hidden state ${model.dim}, every bar read in order` },
    { title: "Mean over time", detail: `one vector of ${model.dim}` },
    { title: "Linear", detail: `${model.dim} to ${model.classes.length} classes` },
  ];
}

function Figure({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">{label}</div>
      <div className="text-sm font-bold font-mono text-gray-200 mt-0.5">{value}</div>
    </div>
  );
}

export default function ModelDiagram({ model }: { model: ModelCard }) {
  const beat = model.accuracy - model.persistence;
  return (
    <section className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-5">
      <div className="flex items-baseline justify-between gap-3 mb-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-gray-400">
          What produces the risk call
        </h2>
        <span className="text-[11px] font-mono text-gray-600">{model.version}</span>
      </div>

      <div className="flex flex-col items-stretch gap-0">
        {stages(model).map((stage, index) => (
          <div key={stage.title}>
            {index > 0 && (
              <div className="flex justify-center py-1">
                <ArrowDown size={13} className="text-gray-700" />
              </div>
            )}
            <div className="rounded-xl border border-white/[0.07] bg-white/[0.03] px-4 py-2.5">
              <div className="text-sm font-semibold text-gray-200">{stage.title}</div>
              <div className="text-[11px] text-gray-500 font-mono mt-0.5">{stage.detail}</div>
            </div>
          </div>
        ))}
        <div className="flex justify-center py-1">
          <ArrowDown size={13} className="text-gray-700" />
        </div>
        <div className="rounded-xl border border-blue-500/25 bg-blue-500/[0.07] px-4 py-2.5">
          <div className="text-sm font-semibold text-blue-200">
            One of {model.classes.join(" / ")}
          </div>
          <div className="text-[11px] text-gray-500 mt-0.5">
            the spread of daily moves over the next {model.horizon} sessions — not its direction
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5 pt-4 border-t border-white/[0.06]">
        <Figure label="Parameters" value={model.parameters.toLocaleString()} />
        <Figure label="Held out" value={`${(model.accuracy * 100).toFixed(1)}%`} />
        <Figure label="Arithmetic rule" value={`${(model.persistence * 100).toFixed(1)}%`} />
        <Figure label="Windows scored" value={model.evaluated.toLocaleString()} />
      </div>

      <p className="text-[11px] text-gray-600 mt-3 leading-relaxed">
        The honest comparison is the third figure, not chance. Bucketing trailing volatility with one line
        of arithmetic already scores {(model.persistence * 100).toFixed(1)}%, so these{" "}
        {model.parameters.toLocaleString()} parameters buy {(beat * 100).toFixed(1)} points. Trained to step{" "}
        {model.steps.toLocaleString()}; everything from {model.cutoff} onward was held out.
      </p>
    </section>
  );
}
