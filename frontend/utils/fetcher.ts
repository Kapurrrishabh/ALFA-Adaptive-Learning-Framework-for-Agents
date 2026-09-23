export const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

// The chat socket lives on the same origin as the routes, so it is derived rather than configured twice.
export const SOCKET_URL = `${API_BASE.replace(/^http/, "ws")}/chat`;

const SESSION_KEY = "alfa.session";

export interface Session {
  token: string;
  email: string;
}

// The backend hands back a token and nothing else, so the email is kept here from what was typed at
// login. localStorage because this runs against a loopback server a person started themselves.
export const session = {
  read(): Session | null {
    if (typeof window === "undefined") return null;
    const stored = window.localStorage.getItem(SESSION_KEY);
    return stored ? (JSON.parse(stored) as Session) : null;
  },
  write(open: Session) {
    window.localStorage.setItem(SESSION_KEY, JSON.stringify(open));
    window.dispatchEvent(new Event(SESSION_EVENT));
  },
  clear() {
    window.localStorage.removeItem(SESSION_KEY);
    window.dispatchEvent(new Event(SESSION_EVENT));
  },
};

// localStorage fires no event in the tab that wrote it, so signing in would leave the navbar stale.
export const SESSION_EVENT = "alfa.session.changed";

export async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  const open = session.read();
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      ...(open ? { Authorization: `Bearer ${open.token}` } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const read = await response.json().catch(() => null);
  if (!response.ok) {
    // The server states what was wrong and which value caused it; passing that through beats a generic
    // failure, and a 401 is how the UI learns the stored token is spent.
    const detail = read && typeof read.detail === "string" ? read.detail : response.statusText;
    throw Object.assign(new Error(`${path}: ${detail}`), { status: response.status });
  }
  return read as T;
}

export const fetcher = <T,>(path: string) => call<T>("GET", path);
