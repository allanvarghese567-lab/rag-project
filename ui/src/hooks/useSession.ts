import { useCallback, useState } from "react";
import type { SessionConfig } from "../types";

const STORAGE_KEY = "rc";

const defaults: SessionConfig = {
  apiBase: "",
  owner: "",
  repo: "",
  token: "",
  branch: "main",
};

function load(): SessionConfig {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...defaults };
    return { ...defaults, ...JSON.parse(raw) };
  } catch {
    return { ...defaults };
  }
}

export function useSession() {
  const [session, setSessionState] = useState<SessionConfig>(load);

  const setSession = useCallback((patch: Partial<SessionConfig>) => {
    setSessionState((prev) => {
      const next = { ...prev, ...patch };
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const save = useCallback(() => {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  }, [session]);

  return { session, setSession, save };
}
