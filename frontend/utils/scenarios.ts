import { fetcher } from "@/utils/fetcher";

/** What `backend/models/scenarios.py` serves: price quantiles per step ahead, a few whole paths, which model
 *  drew them with what it measured, and the sentence saying these are ranges and not calls. */
export interface Scenarios {
  symbol: string;
  as_of: string;
  close: number;
  steps: number;
  paths: number;
  fan: Record<"0.05" | "0.25" | "0.5" | "0.75" | "0.95", number[]>;
  examples: number[][];
  model: {
    name: "generative" | "garch";
    test_from: string;
    measured: { vs_garch: number; vs_garch_error: number; scored: number };
  };
  direction: string;
}

export const readScenarios = (symbol: string, steps = 20) =>
  fetcher<Scenarios>(`/instruments/${symbol}/scenarios?steps=${steps}&paths=200`);
