import { useState } from "react";
import { Card } from "./Card";
import { Button } from "./Button";
import type { JobStatus } from "../types";

interface JobsPanelProps {
  hasApi: boolean;
  hasToken: boolean;
  job: JobStatus | null;
  onApiReindex: (runEvals: boolean) => void;
  onGithubDispatch: (runEvals: boolean) => void;
}

/** Trigger reindex via API or GitHub workflow_dispatch. */
export function JobsPanel({
  hasApi,
  hasToken,
  job,
  onApiReindex,
  onGithubDispatch,
}: JobsPanelProps) {
  const [runEvals, setRunEvals] = useState(false);

  return (
    <Card title="Jobs — temporary reindex runner">
      <p className="muted">
        Calls <code>POST /jobs/reindex</code> on your API (server needs{" "}
        <code>GITHUB_TOKEN</code>). Or dispatch with a GitHub PAT.
      </p>
      <label
        className="row"
        style={{ margin: "8px 0", display: "flex", gap: 6, alignItems: "center" }}
      >
        <input
          type="checkbox"
          checked={runEvals}
          onChange={(e) => setRunEvals(e.target.checked)}
        />
        Also run evals after index
      </label>
      <div className="row">
        <Button primary disabled={!hasApi} onClick={() => onApiReindex(runEvals)}>
          Start reindex via API
        </Button>
        <Button disabled={!hasToken} onClick={() => onGithubDispatch(runEvals)}>
          Dispatch via GitHub token
        </Button>
      </div>
      {job && (
        <div style={{ marginTop: 12 }} className="muted">
          Status: <b>{job.status || "?"}</b>
          {job.conclusion ? ` / ${job.conclusion}` : ""}
          {job.run_id != null ? ` · run #${job.run_number ?? job.run_id}` : ""}
          <br />
          {job.html_url && (
            <a href={job.html_url} target="_blank" rel="noopener noreferrer">
              Open Actions run
            </a>
          )}
        </div>
      )}
    </Card>
  );
}
