"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Briefcase, Loader2, Plus } from "lucide-react";
import toast from "react-hot-toast";

import { useAuth } from "@/hooks/useAuth";
import { call, fetcher } from "@/utils/fetcher";

interface Position {
  symbol: string;
  quantity: number;
  average_price: number;
}

const SIDES = ["buy", "sell"] as const;

export default function PortfolioPage() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const [positions, setPositions] = useState<Position[] | null>(null);
  const [symbol, setSymbol] = useState("");
  const [side, setSide] = useState<(typeof SIDES)[number]>("buy");
  const [quantity, setQuantity] = useState("");
  const [price, setPrice] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!loading && !user) router.push("/login");
  }, [loading, user, router]);

  const load = useCallback(() => {
    fetcher<Position[]>("/portfolio")
      .then(setPositions)
      .catch((err: Error) => toast.error(err.message));
  }, []);

  useEffect(() => {
    if (user) load();
  }, [user, load]);

  const record = async () => {
    try {
      setSaving(true);
      await call("POST", "/trades", {
        symbol: symbol.trim().toUpperCase(),
        side,
        quantity: Number(quantity),
        price: Number(price),
      });
      toast.success(`Recorded ${side} ${quantity} ${symbol.toUpperCase()}`);
      setSymbol("");
      setQuantity("");
      setPrice("");
      load();
    } catch (err) {
      // An over-sell comes back as the server's own sentence, naming the quantity that caused it.
      toast.error(err instanceof Error ? err.message : "The trade was refused.");
    } finally {
      setSaving(false);
    }
  };

  if (loading || !user) return null;
  const ready = symbol.trim() && Number(quantity) > 0 && Number(price) > 0;

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

      <div className="relative z-10 max-w-4xl mx-auto px-4 pt-28 pb-20">
        <header className="mb-8">
          <h1 className="text-2xl font-black tracking-tight flex items-center gap-2.5">
            <Briefcase size={22} className="text-blue-400" /> Portfolio
          </h1>
          <p className="text-sm text-gray-500 mt-1">
            Simulated trades only, folded from the log at average cost. No money moves and no live price is
            fetched, so there is nothing here to mark to market.
          </p>
        </header>

        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-2xl border border-white/[0.06] bg-white/[0.015] overflow-hidden mb-8"
        >
          {positions === null ? (
            <div className="flex items-center gap-2 px-5 py-8 text-sm text-gray-500">
              <Loader2 size={14} className="animate-spin" /> loading the trade log…
            </div>
          ) : positions.length === 0 ? (
            <div className="px-5 py-8 text-sm text-gray-500">
              No open position. Record a trade below and it will fold into one.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[10px] uppercase tracking-wider text-gray-500 border-b border-white/[0.06]">
                  <th className="text-left font-bold px-5 py-3">Symbol</th>
                  <th className="text-right font-bold px-5 py-3">Quantity</th>
                  <th className="text-right font-bold px-5 py-3">Average cost</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => (
                  <tr key={position.symbol} className="border-b border-white/[0.03] last:border-0">
                    <td className="px-5 py-3 font-semibold text-gray-100">{position.symbol}</td>
                    <td className="px-5 py-3 text-right font-mono text-gray-300">{position.quantity}</td>
                    <td className="px-5 py-3 text-right font-mono text-gray-300">
                      {position.average_price.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </motion.div>

        <div className="rounded-2xl border border-white/[0.06] bg-white/[0.015] p-5">
          <h2 className="text-[11px] uppercase tracking-wider font-bold text-gray-400 mb-4">
            Record a simulated trade
          </h2>
          <div className="grid sm:grid-cols-4 gap-3">
            <input
              value={symbol}
              onChange={(event) => setSymbol(event.target.value)}
              placeholder="AAPL"
              className="px-4 py-3 rounded-xl bg-white/[0.03] border border-white/10 focus:border-blue-500/50 outline-none transition-colors text-sm placeholder-gray-600 uppercase"
            />
            <div className="flex gap-1 p-1 rounded-xl bg-white/[0.03] border border-white/10">
              {SIDES.map((one) => (
                <button
                  key={one}
                  onClick={() => setSide(one)}
                  className={`flex-1 rounded-lg text-sm font-semibold capitalize transition-colors ${
                    side === one ? "bg-white/[0.09] text-white" : "text-gray-500 hover:text-gray-300"
                  }`}
                >
                  {one}
                </button>
              ))}
            </div>
            <input
              value={quantity}
              onChange={(event) => setQuantity(event.target.value)}
              placeholder="quantity"
              inputMode="decimal"
              className="px-4 py-3 rounded-xl bg-white/[0.03] border border-white/10 focus:border-blue-500/50 outline-none transition-colors text-sm placeholder-gray-600"
            />
            <input
              value={price}
              onChange={(event) => setPrice(event.target.value)}
              placeholder="price"
              inputMode="decimal"
              className="px-4 py-3 rounded-xl bg-white/[0.03] border border-white/10 focus:border-blue-500/50 outline-none transition-colors text-sm placeholder-gray-600"
            />
          </div>
          <button
            onClick={record}
            disabled={!ready || saving}
            className="mt-4 w-full sm:w-auto px-6 py-3 rounded-xl font-semibold bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white text-sm transition-all flex items-center justify-center gap-2 disabled:opacity-40"
          >
            {saving ? <Loader2 size={15} className="animate-spin" /> : <Plus size={15} />} Record trade
          </button>
        </div>
      </div>
    </main>
  );
}
