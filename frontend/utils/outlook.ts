import { fetcher } from "@/utils/fetcher";

/** What `advisor.describe()` puts in the payload: what is answering and what it scored. Every figure the
 *  panels show comes from here rather than being typed into the page, so a retrained head moves them all. */
export interface ModelCard {
  serving: string;
  version: string;
  parameters: number;
  dim?: number;
  window: number;
  channels: number;
  classes: string[];
  horizon: number;
  edges: number[];
  accuracy: number;
  persistence: number;
  majority: number;
  evaluated: number;
  steps: number;
  cutoff: string;
  calibrated: boolean;
}

export interface Band {
  name: string;
  sigma: number;
  low: number;
  high: number;
  open: boolean;
}

export interface Outlook {
  symbol: string;
  as_of: string;
  dates: string[];
  closes: number[];
  probabilities: number[] | null;
  horizon: number;
  bands: Band[];
  model: ModelCard;
  direction: { accuracy: number; baseline: number };
}

export const readOutlook = (symbol: string) => fetcher<Outlook>(`/instruments/${symbol}/outlook`);

/** Index of the class the head picked. -1 when the instrument was short of the history a head needs,
 *  which is when the evidence carries no outlook either and the panels say so. */
export const chosenClass = (probabilities: number[] | null) =>
  probabilities ? probabilities.indexOf(Math.max(...probabilities)) : -1;
