"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity, AlertTriangle, BadgeCheck, BadgeX, ChevronDown, Loader2, Send, ShieldOff, Sparkles,
} from "lucide-react";
import toast from "react-hot-toast";

import { useAuth } from "@/hooks/useAuth";
import { SOCKET_URL, fetcher, session } from "@/utils/fetcher";

interface Turn {
  question: string;
  asked: string;
  ticker: string;
  intent: string;
  served: string;
  evidence: string;
  spoke: boolean;
  because: string;
  as_of: string;
  margin: number | null;
  confidence: number | null;
  stated: number | null;
  unsupported: string[];
  wrote: string | null;
}

interface Stored {
  message: number;
  judged: boolean | null;
  why: string;
  labeller: string | null;
  answer_cut: number;
  moved: boolean;
  learned: boolean;
}

interface Exchange {
  question: string;
  turn?: Turn;
  stored?: Stored;
}

interface Model {
  generator: string;
  outlook: string | null;
  routing_cut: number;
  answer_cut: number;
}

const SUGGESTIONS = [
  "how has AAPL been doing over the last month ?",
  "is NVDA overbought right now ?",
  "how volatile is TSLA at the moment ?",
  "should i buy MSFT ?",
];

/** The evidence passage as the name/value lines it already is, so a figure can be read against the answer. */
function facts(evidence: string) {
  return evidence
    .split(" ; ")
    .map((line) => {
      const [name, ...rest] = line.trim().split(/\s+/);
      return { name: name.replace(/_/g, " "), value: rest.join(" ") };
    })
    .filter((fact) => fact.value);
}

function EvidencePanel({ turn }: { turn: Turn }) {
  const [open, setOpen] = useState(true);
  const shown = facts(turn.evidence);
  if (!shown.length) return null;

  return (
    <div className="mt-3 rounded-xl border border-white/[0.06] bg-white/[0.02] overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-[11px] uppercase font-bold tracking-wider text-gray-400 hover:text-gray-200 transition-colors"
      >
        <span className="flex items-center gap-2">
          <Activity size={13} /> Evidence it was given
          {turn.as_of && <span className="text-gray-600 normal-case font-medium">as of {turn.as_of}</span>}
        </span>
        <ChevronDown size={14} className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-px bg-white/[0.04] border-t border-white/[0.06]">
          {shown.map((fact) => (
            <div key={fact.name} className="bg-[#080808] px-3 py-2">
              <div className="text-[10px] uppercase tracking-wider text-gray-500">{fact.name}</div>
              <div className="text-sm font-semibold text-gray-100 font-mono">{fact.value}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Badges({ turn, stored }: { turn: Turn; stored?: Stored }) {
  const chips: string[] = [];
  if (turn.intent) chips.push(turn.intent.replace(/_/g, " "));
  if (turn.ticker) chips.push(turn.ticker);
  if (turn.confidence !== null) chips.push(`confidence ${turn.confidence.toFixed(3)}`);
  if (turn.margin !== null) chips.push(`routing margin ${turn.margin.toFixed(3)}`);

  return (
    <div className="flex flex-wrap items-center gap-1.5 mt-3">
      {chips.map((chip) => (
        <span key={chip} className="px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.06] text-[10px] font-medium text-gray-400">
          {chip}
        </span>
      ))}
      {stored?.judged === true && (
        <span className="px-2 py-0.5 rounded-md bg-emerald-500/10 border border-emerald-500/20 text-[10px] font-semibold text-emerald-400 flex items-center gap-1">
          <BadgeCheck size={11} /> judged right by {stored.labeller}
        </span>
      )}
      {stored?.judged === false && (
        <span className="px-2 py-0.5 rounded-md bg-rose-500/10 border border-rose-500/20 text-[10px] font-semibold text-rose-400 flex items-center gap-1">
          <BadgeX size={11} /> judged wrong by {stored.labeller}
        </span>
      )}
      {stored?.judged === null && (
        <span className="px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.06] text-[10px] font-medium text-gray-500">
          unjudged
        </span>
      )}
      {stored?.moved && (
        <span className="px-2 py-0.5 rounded-md bg-blue-500/10 border border-blue-500/20 text-[10px] font-semibold text-blue-400 flex items-center gap-1">
          <Sparkles size={11} /> answer cut now {stored.answer_cut.toFixed(5)}
        </span>
      )}
    </div>
  );
}

function Reply({ exchange }: { exchange: Exchange }) {
  const { turn, stored } = exchange;
  if (!turn) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500 px-4 py-3">
        <Loader2 size={14} className="animate-spin" /> reading the evidence…
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-white/[0.015] p-4">
      {turn.spoke ? (
        <p className="text-[15px] leading-relaxed text-gray-100">{turn.served}</p>
      ) : (
        <div className="flex items-start gap-2.5">
          <ShieldOff size={16} className="text-amber-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-[15px] text-amber-200 font-medium">Withheld — {turn.because}</p>
            {/* The abstention is the result, not an error: it is the only thing in the system that learns. */}
            <p className="text-xs text-gray-500 mt-1">
              The agent had an answer but its confidence sat under the cut it fitted from past feedback.
            </p>
          </div>
        </div>
      )}

      {turn.wrote && (
        <div className="mt-3 flex items-start gap-2.5 rounded-xl border border-amber-500/20 bg-amber-500/[0.04] px-3 py-2">
          <AlertTriangle size={14} className="text-amber-400 mt-0.5 shrink-0" />
          <div className="text-xs text-amber-200/90">
            A guard changed what was served. The model wrote:{" "}
            <span className="font-mono text-amber-100">{turn.wrote}</span>
          </div>
        </div>
      )}

      {turn.unsupported.length > 0 && (
        <div className="mt-2 text-xs text-rose-300/90">
          figures not in the evidence: <span className="font-mono">{turn.unsupported.join(", ")}</span>
        </div>
      )}

      <Badges turn={turn} stored={stored} />
      <EvidencePanel turn={turn} />

      {turn.asked && turn.asked !== turn.question && (
        <p className="mt-3 text-[11px] text-gray-600">
          routed and rewritten into a phrasing the model trained on:{" "}
          <span className="font-mono text-gray-500">{turn.asked}</span>
        </p>
      )}
    </div>
  );
}

export default function ChatPage() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState<"connecting" | "ready" | "working" | "closed">("connecting");
  const [model, setModel] = useState<Model | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const bottom = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!loading && !user) router.push("/login");
  }, [loading, user, router]);

  useEffect(() => {
    if (!user) return;
    fetcher<Model>("/model").then(setModel).catch(() => setModel(null));
  }, [user]);

  useEffect(() => {
    const open = session.read();
    if (!open) return;

    const live = new WebSocket(SOCKET_URL);
    socket.current = live;
    live.onopen = () => live.send(JSON.stringify({ token: open.token, subject: "browser" }));
    live.onclose = () => setStatus("closed");
    live.onerror = () => setStatus("closed");
    live.onmessage = (event) => {
      const frame = JSON.parse(event.data);
      if (frame.stage === "ready") {
        setStatus("ready");
      } else if (frame.stage === "working") {
        setStatus("working");
        setExchanges((was) => [...was, { question: frame.question }]);
      } else if (frame.stage === "turn") {
        setExchanges((was) => was.map((one, i) => (i === was.length - 1 ? { ...one, turn: frame } : one)));
      } else if (frame.stage === "stored") {
        setStatus("ready");
        setExchanges((was) => was.map((one, i) => (i === was.length - 1 ? { ...one, stored: frame } : one)));
        setModel((was) => (was ? { ...was, answer_cut: frame.answer_cut } : was));
        if (frame.moved) toast.success(`The answer cut moved to ${frame.answer_cut.toFixed(5)}`);
      } else if (frame.stage === "refused") {
        setStatus("ready");
        toast.error(frame.because);
      }
    };
    return () => live.close();
  }, [user]);

  // Braced on purpose: a smooth scrollIntoView resolves a promise, and returning it makes React
  // treat it as the cleanup function and tear the page down.
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [exchanges, status]);

  const ask = useCallback(
    (text: string) => {
      const asked = text.trim();
      if (!asked || status !== "ready" || !socket.current) return;
      socket.current.send(JSON.stringify({ question: asked }));
      setQuestion("");
    },
    [status],
  );

  if (loading || !user) return null;

  return (
    <main className="relative min-h-screen bg-[#050505] text-white">
      <div
        className="absolute inset-0 z-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(#ffffff 1px, transparent 1px), linear-gradient(90deg, #ffffff 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      <div className="relative z-10 max-w-3xl mx-auto px-4 pt-28 pb-40">
        <header className="mb-8">
          <h1 className="text-2xl font-black tracking-tight">Ask the agent</h1>
          <p className="text-sm text-gray-500 mt-1">
            Every figure in an answer is one the evidence below it states. When the agent is not confident
            enough, it says so instead — and that threshold is the part that learns.
          </p>
          {model && (
            <div className="flex flex-wrap gap-1.5 mt-3 text-[10px] font-medium">
              <span className="px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.06] text-gray-400">
                generator {model.generator}
              </span>
              {model.outlook && (
                <span className="px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.06] text-gray-400">
                  outlook {model.outlook}
                </span>
              )}
              <span className="px-2 py-0.5 rounded-md bg-white/[0.04] border border-white/[0.06] text-gray-400">
                routing cut {model.routing_cut.toFixed(3)}
              </span>
              <span className="px-2 py-0.5 rounded-md bg-blue-500/10 border border-blue-500/20 text-blue-400">
                answer cut {model.answer_cut.toFixed(5)}
              </span>
            </div>
          )}
        </header>

        {exchanges.length === 0 && (
          <div className="grid sm:grid-cols-2 gap-2">
            {SUGGESTIONS.map((one) => (
              <button
                key={one}
                onClick={() => ask(one)}
                disabled={status !== "ready"}
                className="text-left px-4 py-3 rounded-xl border border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.04] transition-colors text-sm text-gray-300 disabled:opacity-40"
              >
                {one}
              </button>
            ))}
          </div>
        )}

        <div className="space-y-6">
          <AnimatePresence initial={false}>
            {exchanges.map((exchange, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-2"
              >
                <p className="text-sm font-semibold text-blue-300 ml-1">{exchange.question}</p>
                <Reply exchange={exchange} />
              </motion.div>
            ))}
          </AnimatePresence>
          <div ref={bottom} />
        </div>
      </div>

      <div className="fixed bottom-0 left-0 right-0 z-20 bg-gradient-to-t from-[#050505] via-[#050505] to-transparent pt-8 pb-6">
        <div className="max-w-3xl mx-auto px-4">
          <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-2 focus-within:border-blue-500/50 transition-colors">
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && ask(question)}
              placeholder={status === "closed" ? "the connection closed — reload to reopen it" : "how has AAPL been doing ?"}
              disabled={status === "closed"}
              className="flex-1 bg-transparent outline-none py-2 text-[15px] placeholder-gray-600 disabled:opacity-50"
            />
            <button
              onClick={() => ask(question)}
              disabled={status !== "ready" || !question.trim()}
              className="p-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-30 disabled:hover:bg-blue-600 transition-colors"
            >
              {status === "working" ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
            </button>
          </div>
          <p className="text-[10px] text-gray-600 mt-2 ml-1">
            {status === "connecting" && "opening the socket…"}
            {status === "ready" && "connected — answers are simulated from historical prices, never advice to trade"}
            {status === "working" && "the decoder is running"}
            {status === "closed" && "disconnected"}
          </p>
        </div>
      </div>
    </main>
  );
}
