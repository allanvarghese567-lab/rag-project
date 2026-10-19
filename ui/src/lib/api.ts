/** Shared HTTP helpers for the live RAG API and optional GitHub calls. */

function formatError(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((x) =>
        typeof x === "object" && x && "msg" in x
          ? String((x as { msg: string }).msg)
          : JSON.stringify(x)
      )
      .join("; ");
  }
  return fallback;
}

export async function apiFetch<
  T = unknown
>(
  apiBase: string,
  path: string,
  opts: RequestInit = {}
): Promise<T> {
  const base = apiBase.replace(/\/$/, "");
  if (!base) throw new Error("Set API base URL under Connect");

  const headers: Record<string, string> = {
    ...((opts.headers as Record<string, string>) || {}),
  };
  if (opts.body && !(opts.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(base + path, { ...opts, headers });
  const text = await res.text();
  let json: Record<string, unknown> = {};
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    json = { raw: text };
  }

  if (!res.ok) {
    const msg = formatError(
      json.detail,
      String(json.message || text || `HTTP ${res.status}`)
    );
    throw new Error(msg);
  }
  return json as T;
}

export async function githubFetch<
  T = unknown
>(
  owner: string,
  repo: string,
  token: string,
  path: string,
  opts: RequestInit = {}
): Promise<T> {
  const res = await fetch(`https://api.github.com/repos/${owner}/${repo}${path}`, {
    ...opts,
    cache: "no-cache",
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      ...(opts.body ? { "Content-Type": "application/json" } : {}),
      ...((opts.headers as Record<string, string>) || {}),
    },
  });
  const text = await res.text();
  let json: Record<string, unknown> = {};
  try {
    json = text ? JSON.parse(text) : {};
  } catch {
    json = { message: text };
  }
  if (!res.ok) {
    throw new Error(String(json.message || text || res.status));
  }
  return json as T;
}
