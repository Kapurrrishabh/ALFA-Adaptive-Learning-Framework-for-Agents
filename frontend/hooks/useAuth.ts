"use client";

import { useCallback, useEffect, useState } from "react";

import { SESSION_EVENT, Session, call, session } from "@/utils/fetcher";

interface UseAuthReturn {
  user: Session | null;
  loading: boolean;
  signOut: () => Promise<void>;
}

export function useAuth(): UseAuthReturn {
  const [user, setUser] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const read = () => {
      setUser(session.read());
      setLoading(false);
    };
    read();
    // `storage` covers another tab, the custom event covers this one.
    window.addEventListener("storage", read);
    window.addEventListener(SESSION_EVENT, read);
    return () => {
      window.removeEventListener("storage", read);
      window.removeEventListener(SESSION_EVENT, read);
    };
  }, []);

  const signOut = useCallback(async () => {
    // The token is dropped locally whatever the server says: a spent token is already unusable, and
    // leaving it in the browser because logout failed is the worse outcome.
    try {
      await call("POST", "/auth/logout");
    } finally {
      session.clear();
    }
  }, []);

  return { user, loading, signOut };
}
