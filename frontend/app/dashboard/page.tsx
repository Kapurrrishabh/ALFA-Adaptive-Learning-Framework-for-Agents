"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Activity, ArrowRight, Loader2, Search } from "lucide-react";

import ModelDiagram from "@/components/ModelDiagram";
import RiskCall from "@/components/RiskCall";
import { useAuth } from "@/hooks/useAuth";
import { fetcher } from "@/utils/fetcher";
import { readOutlook, type Outlook } from "@/utils/outlook";

interface Snapshot {
  symbol: string;
  as_of: string;
  facts: Record<string, string>;
  evidence: string;
}

// The names advisory.as_text produces, in reading order, with the wording a person uses for each. Only
// these are shown: a fact the snapshot gains would otherwise appear here unlabelled and unexplained.
const FACTS: { name: string; label: string; note: string }[] = [
  { name: "close", label: "Last close", note: "the price the figures below are computed against" },
  { name: "return_20d", label: "20 day move", note: "how the price has changed over 20 sessions" },
  { name: "return_5d", label: "5 day move", note: "the same over the last week of sessions" },
  { name: "volatility_20d", label: "Volatility", note: "annualised from the last 20 daily moves" },
  { name: "rsi_14", label: "RSI (14)", note: "over 70 is called overbought, under 30 oversold" },
  { name: "drawdown_60d", label: "Drawdown", note: "the fall from the highest close in 60 sessions" },
  { name: "sma_20", label: "20 day average", note: "above it is the usual short-term uptrend read" },
  { name: "sma_50", label: "50 day average", note: "the slower trend the 20 day is read against" },
  { name: "atr_14", label: "Average range", note: "the typical daily swing, in price not percent" },
  { name: "outlook", label: "Week-ahead risk", note: "the price head's own call, not the decoder's" },
  { name: "outlook_confidence", label: "Its confidence", note: "how sure that head is of the call" },
];

/** Green for a gain, red for a fall, plain for everything else. Read off the sign the backend formatted. */
function tone(value: string) {
  if (value.startsWith("+")) return "text-emerald-400";
  if (value.startsWith("-")) return "text-rose-400";
  return "text-gray-100";
}

function Fact({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
      <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">{label}</div>
      <div className={`text-xl font-bold font-mono mt-0.5 ${tone(value)}`}>{value}</div>
      <div className="text-[11px] text-gray-600 mt-1 leading-snug">{note}</div>
    </div>
  );
}

export default function DashboardPage() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const [universe, setUniverse] = useState<string[]>([]);
  const [chosen, setChosen] = useState("");
  const [typed, setTyped] = useState("");
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [outlook, setOutlook] = useState<Outlook | null>(null);
  const [failed, setFailed] = useState("");
  // Kept apart from `failed`: with no head loaded the route says so and the snapshot is still worth
  // showing, so the panels carry the reason rather than the page going blank.
  const [noOutlook, setNoOutlook] = useState("");

  useEffect(() => {
    if (!loading && !user) router.push("/login");
  }, [loading, user, router]);

  useEffect(() => {
    if (!user) return;
    fetcher<string[]>("/instruments")
      .then((held) => { setUniverse(held); setChosen((was) => was || held[0] || ""); })
      .catch((error: Error) => setFailed(error.message));
  }, [user]);

  useEffect(() => {
    if (!chosen) return;
    setSnapshot(null);
    setOutlook(null);
    setFailed("");
    setNoOutlook("");
    fetcher<Snapshot>(`/instruments/${chosen}`)
      .then(setSnapshot)
      .catch((error: Error) => setFailed(error.message));
    readOutlook(chosen)
      .then(setOutlook)
      .catch((error: Error) => setNoOutlook(error.message));
  }, [chosen]);

  const matching = useMemo(() => {
    const term = typed.trim().toUpperCase();
    return term ? universe.filter((one) => one.includes(term)).slice(0, 8) : [];
  }, [typed, universe]);

  const ask = useCallback(() => {
    // A whole question rather than the ticker alone: the router places this shape, and a bare symbol is
    // the "unclear question" refusal the queue has open.
    router.push(`/?ask=${encodeURIComponent(`how has ${chosen} been doing over the last month ?`)}`);
  }, [router, chosen]);

  if (loading || !user) return null;

  return (
    <main className="relative min-h-screen bg-[#050505] text-white">
      <div className="relative z-10 max-w-5xl mx-auto px-4 pt-28 pb-24">
        <header className="mb-6">
          <h1 className="text-2xl font-black tracking-tight">Instruments</h1>
          <p className="text-sm text-gray-500 mt-1">
            Exactly the snapshot the agent is handed when you ask about one of these — the same figures,
            read off the same price files, cut off at the same date.
          </p>
        </header>

        <div className="relative mb-5">
          <div className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-3 focus-within:border-blue-500/50 transition-colors">
            <Search size={15} className="text-gray-500" />
            <input
              value={typed}
              onChange={(event) => setTyped(event.target.value)}
              placeholder={`search ${universe.length} instruments`}
              className="flex-1 bg-transparent outline-none py-2.5 text-sm placeholder-gray-600"
            />
          </div>
          {matching.length > 0 && (
            <div className="absolute z-20 mt-1 w-full rounded-xl border border-white/[0.09] bg-[#0c0c0c] overflow-hidden">
              {matching.map((one) => (
                <button
                  key={one}
                  onClick={() => { setChosen(one); setTyped(""); }}
                  className="w-full text-left px-4 py-2.5 text-sm text-gray-300 hover:bg-white/[0.05] font-mono"
                >
                  {one}
                </button>
              ))}
            </div>
          )}
        </div>

        {failed && (
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/[0.05] px-4 py-3 text-sm text-rose-300">
            {failed}
          </div>
        )}

        {!snapshot && !failed && (
          <div className="flex items-center gap-2 text-sm text-gray-500 py-8">
            <Loader2 size={15} className="animate-spin" /> reading the price file…
          </div>
        )}

        {snapshot && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
            <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
              <div>
                <div className="flex items-center gap-2">
                  <Activity size={18} className="text-blue-400" />
                  <span className="text-3xl font-black font-mono tracking-tight">{snapshot.symbol}</span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  as of {snapshot.as_of} — the last bar on file, so nothing here is look-ahead
                </div>
              </div>
              <button
                onClick={ask}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-sm font-semibold transition-colors"
              >
                Ask about {snapshot.symbol} <ArrowRight size={15} />
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2">
              {FACTS.filter((fact) => snapshot.facts[fact.name] !== undefined).map((fact) => (
                <Fact key={fact.name} label={fact.label} value={snapshot.facts[fact.name]} note={fact.note} />
              ))}
            </div>

            {outlook && (
              <div className="grid lg:grid-cols-5 gap-3 mt-3">
                <div className="lg:col-span-3">
                  <RiskCall drawn={outlook} stated={snapshot.facts.outlook_confidence} />
                </div>
                <div className="lg:col-span-2">
                  <ModelDiagram model={outlook.model} />
                </div>
              </div>
            )}

            {noOutlook && (
              <div className="mt-3 rounded-xl border border-amber-500/20 bg-amber-500/[0.05] px-4 py-3 text-xs text-amber-300">
                no week-ahead panel: {noOutlook}
              </div>
            )}

            <div className="mt-5 rounded-xl border border-white/[0.06] bg-white/[0.015] px-4 py-3">
              <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold mb-1.5">
                The passage the agent reads
              </div>
              <p className="text-xs font-mono text-gray-400 leading-relaxed break-words">{snapshot.evidence}</p>
              <p className="text-[11px] text-gray-600 mt-2">
                Every figure in an answer about {snapshot.symbol} has to appear in this line, and the guard
                replaces the answer when one does not.
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </main>
  );
}
