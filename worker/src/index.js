/**
 * Webhook receiver + live fan-out (Cloudflare Worker + Durable Object).
 *
 *   GitHub --POST /webhook (HMAC verified)--> Worker --> Hub (Durable Object)
 *   UI     --WebSocket /ws?key=VIEW_KEY-----> Worker --> Hub  (events pushed live)
 *
 * Secrets: WEBHOOK_SECRET (same as the GitHub webhook secret), VIEW_KEY (UI password).
 */

/** Verify GitHub's X-Hub-Signature-256 header (constant-time compare via WebCrypto). */
async function verify(body, sig, secret) {
  if (!secret || !/^sha256=[0-9a-f]{64}$/.test(sig || "")) return false;
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["verify"]);
  const bytes = Uint8Array.from(sig.slice(7).match(/../g).map((h) => parseInt(h, 16)));
  return crypto.subtle.verify("HMAC", key, bytes, enc.encode(body));
}

/** Reduce a GitHub payload to a small event the UI can show. */
function summarize(event, p, id) {
  const base = { id, event, action: p.action || "", ts: Date.now(),
                 repo: p.repository?.full_name || "", sender: p.sender?.login || "" };
  if (event === "workflow_run") {
    const r = p.workflow_run;
    return { ...base, summary: `${r.name} #${r.run_number} ${r.status}${r.conclusion ? " / " + r.conclusion : ""}`, url: r.html_url };
  }
  if (event === "workflow_job") {
    const j = p.workflow_job;
    return { ...base, summary: `${j.name} ${j.status}${j.conclusion ? " / " + j.conclusion : ""}`, url: j.html_url };
  }
  if (event === "push") {
    return { ...base, summary: `${p.ref} (${(p.commits || []).length} commits)` };
  }
  return { ...base, summary: p.zen || "" };  // ping and others
}

export default {
  /** Routes: POST /webhook from GitHub, GET /ws (WebSocket) from the UI. */
  async fetch(req, env) {
    const url = new URL(req.url);
    const hub = env.HUB.get(env.HUB.idFromName("main"));

    if (url.pathname === "/webhook" && req.method === "POST") {
      const body = await req.text();
      if (!(await verify(body, req.headers.get("x-hub-signature-256"), env.WEBHOOK_SECRET)))
        return new Response("bad signature", { status: 401 });
      const ev = summarize(req.headers.get("x-github-event"), JSON.parse(body),
                           req.headers.get("x-github-delivery"));
      await hub.fetch("https://hub/event", { method: "POST", body: JSON.stringify(ev) });
      return new Response("ok");
    }

    if (url.pathname === "/ws") {
      if (req.headers.get("Upgrade") !== "websocket") return new Response("expected websocket", { status: 426 });
      if (!env.VIEW_KEY || url.searchParams.get("key") !== env.VIEW_KEY)
        return new Response("forbidden", { status: 403 });
      return hub.fetch(req);
    }
    return new Response("rag-webhook-hub", { status: 200 });
  },
};

/** Durable Object: holds the open WebSockets and the last 30 events. */
export class Hub {
  constructor(state) { this.state = state; }

  async fetch(req) {
    const path = new URL(req.url).pathname;
    if (path === "/ws") {                                   // a UI connecting
      const [client, server] = Object.values(new WebSocketPair());
      this.state.acceptWebSocket(server);
      const recent = (await this.state.storage.get("recent")) || [];
      server.send(JSON.stringify({ type: "hello", recent }));
      return new Response(null, { status: 101, webSocket: client });
    }
    if (path === "/event") {                                // a new GitHub event
      const ev = await req.json();
      const recent = ((await this.state.storage.get("recent")) || []).concat(ev).slice(-30);
      await this.state.storage.put("recent", recent);
      for (const ws of this.state.getWebSockets()) ws.send(JSON.stringify({ type: "event", event: ev }));
      return new Response("ok");
    }
    return new Response("not found", { status: 404 });
  }

  webSocketMessage() {}   // the UI only listens
  webSocketClose(ws) { ws.close(); }
}
