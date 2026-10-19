import { useState } from "react";
import { Card } from "./Card";
import { Button } from "./Button";
import { Field, TextArea, TextInput, TextSelect } from "./Field";
import type { AskResponse, LlmsConfig, StatusResponse, TracesResponse } from "../types";

interface LiveApiPanelProps {
  apiBase: string;
  status: StatusResponse | null;
  answer: AskResponse | null;
  traces: TracesResponse | null;
  llmCfg: LlmsConfig | null;
  onHealth: () => void;
  onStatus: () => void;
  onTraces: () => void;
  onAsk: (q: string, backend: string, k: number, llmProfile: string) => void;
  onIngest: (file: File) => void;
}

const BACKENDS = [
  "chroma",
  "numpy",
  "hybrid",
  "chroma+rerank",
  "hybrid+rerank",
] as const;

/** Status / ask / ingest / traces against the live API. */
export function LiveApiPanel({
  apiBase,
  status,
  answer,
  traces,
  llmCfg,
  onHealth,
  onStatus,
  onTraces,
  onAsk,
  onIngest,
}: LiveApiPanelProps) {
  const [question, setQuestion] = useState("");
  const [backend, setBackend] = useState<string>("chroma");
  const [k, setK] = useState(5);
  const [llmProfile, setLlmProfile] = useState("");
  const [file, setFile] = useState<File | null>(null);

  const profiles = llmCfg?.profiles ?? [];
  const activeId =
    llmCfg?.active ??
    (status?.llm_profile && typeof status.llm_profile.id === "string"
      ? status.llm_profile.id
      : "");

  return (
    <>
      <Card title="Live RAG API">
        <p className="muted">
          Base URL: <code>{apiBase || "(not set — open Connect)"}</code>
          {" · "}Server console: open <code>/console</code> on the API host.
        </p>
        <div className="row" style={{ marginTop: 8 }}>
          <Button primary onClick={onStatus}>
            Refresh status
          </Button>
          <Button onClick={onTraces}>Load traces</Button>
          <Button onClick={onHealth}>Health</Button>
        </div>
        {status && (
          <div style={{ marginTop: 10 }} className="muted">
            Chunks: <b>{status.chunks}</b> · Sources:{" "}
            {status.sources?.length ? status.sources.join(", ") : "none"}
            <br />
            Embedding: <code>{String(status.embedding_profile?.id ?? "?")}</code>
            <br />
            LLM: <code>{String(status.llm_profile?.id ?? "?")}</code> ·{" "}
            {String(status.llm_profile?.provider ?? "")} /{" "}
            {String(status.llm_profile?.model ?? "")}
          </div>
        )}
      </Card>

      <Card title="Ask">
        <Field label="Question">
          <TextArea
            rows={3}
            placeholder="What does the document say about …?"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
        </Field>
        <div className="row">
          <div className="grow">
            <Field label="Backend">
              <TextSelect
                value={backend}
                onChange={(e) => setBackend(e.target.value)}
              >
                {BACKENDS.map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </TextSelect>
            </Field>
          </div>
          <div>
            <Field label="k">
              <TextInput
                type="number"
                min={1}
                max={50}
                style={{ width: 72 }}
                value={k}
                onChange={(e) => setK(Number(e.target.value) || 5)}
              />
            </Field>
          </div>
        </div>
        <Field label="LLM profile">
          <TextSelect
            value={llmProfile}
            onChange={(e) => setLlmProfile(e.target.value)}
          >
            <option value="">(active: {activeId || "default"})</option>
            {profiles.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} — {p.provider} / {p.model}
              </option>
            ))}
          </TextSelect>
        </Field>
        <div className="row" style={{ marginTop: 10 }}>
          <Button primary onClick={() => onAsk(question, backend, k, llmProfile)}>
            Ask
          </Button>
        </div>
        {answer && (
          <div style={{ marginTop: 12 }}>
            <b>Answer</b>
            <pre className="answer-box">{answer.answer}</pre>
            <details open>
              <summary className="muted">
                Hits ({answer.hits?.length ?? 0}) · LLM{" "}
                {answer.llm_profile ?? "?"}
              </summary>
              {(answer.hits || []).map((h, i) => (
                <Card key={i} className="" title={undefined}>
                  <div className="muted">
                    {h.source} ·{" "}
                    {h.score != null ? Number(h.score).toFixed(3) : "?"}
                  </div>
                  <div className="hit-preview">{(h.content || "").slice(0, 500)}</div>
                </Card>
              ))}
            </details>
          </div>
        )}
      </Card>

      <Card title="Ingest PDF">
        <Field label="PDF file">
          <TextInput
            type="file"
            accept=".pdf,application/pdf"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </Field>
        <div className="row" style={{ marginTop: 10 }}>
          <Button
            primary
            disabled={!file}
            onClick={() => file && onIngest(file)}
          >
            Upload &amp; index
          </Button>
        </div>
      </Card>

      {traces && (
        <Card title="Recent traces">
          {(traces.traces || []).length === 0 && (
            <p className="muted">No traces</p>
          )}
          {(traces.traces || []).map((t, i) => (
            <Card key={i}>
              <div className="row">
                <b>{t.name}</b>
                <span className="muted grow">{t.total_ms} ms</span>
              </div>
              {(t.spans || []).map((s, j) => (
                <div key={j} className="mono-line">
                  {s.name} {s.ms} ms
                </div>
              ))}
            </Card>
          ))}
        </Card>
      )}
    </>
  );
}
