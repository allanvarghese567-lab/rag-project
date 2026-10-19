import { useCallback, useState } from "react";
import { Nav } from "./components/Nav";
import { Toast } from "./components/Toast";
import { ConnectPanel } from "./components/ConnectPanel";
import { LiveApiPanel } from "./components/LiveApiPanel";
import { LlmsPanel } from "./components/LlmsPanel";
import { EmbeddingsPanel } from "./components/EmbeddingsPanel";
import { JobsPanel } from "./components/JobsPanel";
import { useSession } from "./hooks/useSession";
import { useToast } from "./hooks/useToast";
import { apiFetch, githubFetch } from "./lib/api";
import type {
  AskResponse,
  EmbeddingsConfig,
  JobStatus,
  LlmsConfig,
  StatusResponse,
  TabId,
  TracesResponse,
} from "./types";

export default function App() {
  const { session, setSession, save } = useSession();
  const { message, error, visible, toast } = useToast();

  const [tab, setTab] = useState<TabId>(session.apiBase ? "api" : "conn");
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [traces, setTraces] = useState<TracesResponse | null>(null);
  const [llmCfg, setLlmCfg] = useState<LlmsConfig | null>(null);
  const [embCfg, setEmbCfg] = useState<EmbeddingsConfig | null>(null);
  const [job, setJob] = useState<JobStatus | null>(null);

  const run = useCallback(
    async (fn: () => Promise<void>) => {
      try {
        await fn();
      } catch (e) {
        toast(e instanceof Error ? e.message : String(e), true);
      }
    },
    [toast]
  );

  const handleTab = (id: TabId) => {
    setTab(id);
    if (id === "llms" && !llmCfg && session.apiBase) {
      void run(async () => {
        setLlmCfg(await apiFetch<LlmsConfig>(session.apiBase, "/llms"));
      });
    }
  };

  return (
    <main>
      <Nav active={tab} onChange={handleTab} />
      <div>
        {tab === "conn" && (
          <ConnectPanel
            session={session}
            onChange={setSession}
            onSaveApi={() => {
              save();
              toast("Saved");
              setTab("api");
            }}
            onConnectGithub={() =>
              void run(async () => {
                save();
                await githubFetch(session.owner, session.repo, session.token, "");
                setTab("emb");
                toast("Connected");
              })
            }
          />
        )}

        {tab === "api" && (
          <LiveApiPanel
            apiBase={session.apiBase}
            status={status}
            answer={answer}
            traces={traces}
            llmCfg={llmCfg}
            onHealth={() =>
              void run(async () => {
                const h = await apiFetch<{ status: string }>(
                  session.apiBase,
                  "/health"
                );
                toast(`Health: ${h.status}`);
              })
            }
            onStatus={() =>
              void run(async () => {
                setStatus(
                  await apiFetch<StatusResponse>(session.apiBase, "/status")
                );
                try {
                  setLlmCfg(
                    await apiFetch<LlmsConfig>(session.apiBase, "/llms")
                  );
                } catch {
                  /* optional */
                }
                toast("Status loaded");
              })
            }
            onTraces={() =>
              void run(async () => {
                setTraces(
                  await apiFetch<TracesResponse>(
                    session.apiBase,
                    "/traces?limit=15"
                  )
                );
                toast("Traces loaded");
              })
            }
            onAsk={(q, backend, k, llmProfile) =>
              void run(async () => {
                if (!q.trim()) throw new Error("Enter a question");
                toast("Asking…");
                const body: Record<string, unknown> = {
                  question: q,
                  backend,
                  k,
                };
                if (llmProfile) body.llm_profile = llmProfile;
                setAnswer(
                  await apiFetch<AskResponse>(session.apiBase, "/ask", {
                    method: "POST",
                    body: JSON.stringify(body),
                  })
                );
                toast("Done");
              })
            }
            onIngest={(file) =>
              void run(async () => {
                toast("Uploading…");
                const fd = new FormData();
                fd.append("file", file, file.name);
                const j = await apiFetch<{ file: string; chunks: number }>(
                  session.apiBase,
                  "/ingest",
                  { method: "POST", body: fd }
                );
                toast(`Ingested ${j.file}: ${j.chunks} chunks`);
                try {
                  setStatus(
                    await apiFetch<StatusResponse>(session.apiBase, "/status")
                  );
                } catch {
                  /* optional */
                }
              })
            }
          />
        )}

        {tab === "llms" && (
          <LlmsPanel
            config={llmCfg}
            hasApi={!!session.apiBase}
            onLoad={() =>
              void run(async () => {
                setLlmCfg(
                  await apiFetch<LlmsConfig>(session.apiBase, "/llms")
                );
                toast("LLM profiles loaded");
              })
            }
          />
        )}

        {tab === "emb" && (
          <EmbeddingsPanel
            config={embCfg}
            hasApi={!!session.apiBase}
            onLoad={() =>
              void run(async () => {
                setEmbCfg(
                  await apiFetch<EmbeddingsConfig>(
                    session.apiBase,
                    "/embeddings"
                  )
                );
                toast("Embeddings loaded");
              })
            }
          />
        )}

        {tab === "jobs" && (
          <JobsPanel
            hasApi={!!session.apiBase}
            hasToken={!!session.token}
            job={job}
            onApiReindex={(runEvals) =>
              void run(async () => {
                toast("Dispatching reindex…");
                const j = await apiFetch<JobStatus>(
                  session.apiBase,
                  "/jobs/reindex",
                  {
                    method: "POST",
                    body: JSON.stringify({ run_evals: runEvals }),
                  }
                );
                setJob(j);
                toast(j.html_url ? "Job started" : "Dispatched");
              })
            }
            onGithubDispatch={(runEvals) =>
              void run(async () => {
                await githubFetch(
                  session.owner,
                  session.repo,
                  session.token,
                  "/actions/workflows/reindex.yml/dispatches",
                  {
                    method: "POST",
                    body: JSON.stringify({
                      ref: session.branch || "main",
                      inputs: { run_evals: String(runEvals) },
                    }),
                  }
                );
                toast("Reindex requested");
              })
            }
          />
        )}
      </div>
      <Toast message={message} error={error} visible={visible} />
    </main>
  );
}
