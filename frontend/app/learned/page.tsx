"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Loader2, ShieldOff, Sparkles } from "lucide-react";

import { useAuth } from "@/hooks/useAuth";
import { fetcher } from "@/utils/fetcher";

interface Learned {
  asked: number;
  spoke: number;
  right: number;
  wrong: number;
  unjudged: number;
  withheld: Record<string, number>;
  answer_cut: number;
  routing_cut: number;
  log_rows: number;
  judges_agreed: number;
  judges_compared: number;
}

// The same wording the chat page uses for each refusal, because the same five stages produced them.
const WHY_QUIET: Record<string, string> = {
  "low confidence": "the answer sat under the cut fitted from past feedback",
  "unsupported figure": "the answer stated a figure its evidence does not",
  "unclear question": "no trained question shape matched the wording",
  "no subject": "no company or ticker to gather evidence about",
  missing: "the evidence did not carry a figure that intent has to quote",
};

/** The missing-facts stage names the facts it wanted, so its reason is matched on the stage, not whole. */
function why(because: string) {
  return WHY_QUIET[because.startsWith("missing ") ? "missing" : because] ?? "";
}

function Stat({ label, value, note, tone = "text-gray-100" }: {
  label: string; value: string; note: string; tone?: string;
}) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] px-4 py-3">
      <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">{label}</div>
      <div className={`text-2xl font-bold font-mono mt-0.5 ${tone}`}>{value}</div>
      <div className="text-[11px] text-gray-600 mt-1 leading-snug">{note}</div>
    </div>
  );
}

export default function LearnedPage() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const [learned, setLearned] = useState<Learned | null>(null);
  const [failed, setFailed] = useState("");

  useEffect(() => {
    if (!loading && !user) router.push("/login");
  }, [loading, user, router]);

  useEffect(() => {
    if (!user) return;
    fetcher<Learned>("/learned").then(setLearned).catch((error: Error) => setFailed(error.message));
  }, [user]);

  if (loading || !user) return null;

  const judged = learned ? learned.right + learned.wrong : 0;

  return (
    <main className="relative min-h-screen bg-[#050505] text-white">
      <div className="relative z-10 max-w-4xl mx-auto px-4 pt-28 pb-24">
        <header className="mb-6">
          <h1 className="text-2xl font-black tracking-tight">What your feedback has changed</h1>
          <p className="text-sm text-gray-500 mt-1">
            The foundation model is frozen and never retrained. The only thing that moves is where the
            agent draws the line between answering and staying quiet — and this is that line.
          </p>
        </header>

        {failed && (
          <div className="rounded-xl border border-rose-500/20 bg-rose-500/[0.05] px-4 py-3 text-sm text-rose-300">
            {failed}
          </div>
        )}

        {!learned && !failed && (
          <div className="flex items-center gap-2 text-sm text-gray-500 py-8">
            <Loader2 size={15} className="animate-spin" /> reading the feedback log…
          </div>
        )}

        {learned && (
          <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
            <div className="rounded-2xl border border-blue-500/20 bg-blue-500/[0.04] px-5 py-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-blue-400 font-bold">
                <Sparkles size={13} /> The cut that is learning
              </div>
              <div className="text-4xl font-black font-mono mt-1.5 text-blue-300">
                {learned.answer_cut.toFixed(5)}
              </div>
              <p className="text-xs text-gray-400 mt-2 leading-relaxed">
                Below this self-confidence the agent says nothing. It is solved for from the{" "}
                {learned.log_rows} {learned.log_rows === 1 ? "turn" : "turns"} in the shared feedback log,
                and refitted after every answer. The routing cut, at {learned.routing_cut.toFixed(3)}, is
                fitted on its own measurement instead and does not move here.
              </p>
            </div>

            <div>
              <h2 className="text-sm font-bold text-gray-300 mb-2">Your turns</h2>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <Stat label="Asked" value={String(learned.asked)} note="questions you have put to it" />
                <Stat
                  label="Answered"
                  value={learned.asked ? `${Math.round((100 * learned.spoke) / learned.asked)}%` : "—"}
                  note={`${learned.spoke} spoke, ${learned.asked - learned.spoke} withheld`}
                />
                <Stat
                  label="Judged right"
                  value={judged ? `${Math.round((100 * learned.right) / judged)}%` : "—"}
                  note={`${learned.right} of ${judged} that a judge ruled on`}
                  tone="text-emerald-400"
                />
                <Stat
                  label="Unjudged"
                  value={String(learned.unjudged)}
                  note="no verdict, so they are not in the fit"
                />
              </div>
              {learned.asked > 0 && judged === 0 && (
                <p className="text-[11px] text-gray-600 mt-2">
                  Nothing of yours has been judged yet, so the cut above was fitted entirely from other
                  turns in the log.
                </p>
              )}
            </div>

            {Object.keys(learned.withheld).length > 0 && (
              <div>
                <h2 className="text-sm font-bold text-gray-300 mb-2 flex items-center gap-2">
                  <ShieldOff size={14} className="text-amber-400" /> Which stage stayed quiet
                </h2>
                <div className="rounded-xl border border-white/[0.06] overflow-hidden divide-y divide-white/[0.06]">
                  {Object.entries(learned.withheld).map(([because, count]) => (
                    <div key={because} className="flex items-baseline gap-3 bg-white/[0.02] px-4 py-2.5">
                      <span className="font-mono text-sm text-amber-300 w-6 text-right">{count}</span>
                      <span className="text-sm text-gray-200 font-medium">{because}</span>
                      <span className="text-[11px] text-gray-600">{why(because)}</span>
                    </div>
                  ))}
                </div>
                <p className="text-[11px] text-gray-600 mt-2">
                  A refusal is a result here, not a failure: each of these is a different stage deciding it
                  could not answer without inventing something.
                </p>
              </div>
            )}

            {learned.judges_compared > 0 && (
              <div className="rounded-xl border border-white/[0.06] bg-white/[0.015] px-4 py-3">
                <div className="text-[10px] uppercase tracking-wider text-gray-500 font-bold">
                  Where two judges saw the same answer
                </div>
                <p className="text-sm text-gray-300 mt-1">
                  They agreed on {learned.judges_agreed} of {learned.judges_compared}. A result that only
                  holds under one judge is not a result, so the log records which one ruled on each row.
                </p>
              </div>
            )}
          </motion.div>
        )}
      </div>
    </main>
  );
}
