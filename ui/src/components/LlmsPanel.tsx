import { Card } from "./Card";
import { Button } from "./Button";
import type { LlmsConfig } from "../types";

interface LlmsPanelProps {
  config: LlmsConfig | null;
  hasApi: boolean;
  onLoad: () => void;
}

/** List LLM gateway profiles from the API. */
export function LlmsPanel({ config, hasApi, onLoad }: LlmsPanelProps) {
  if (!config) {
    return (
      <Card>
        <p className="muted">
          Load from live API after setting base URL under Connect.
        </p>
        <div className="row">
          <Button primary disabled={!hasApi} onClick={onLoad}>
            Load from API
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <Card title="LLM profiles">
      <p className="muted">
        Active: <code>{config.active}</code>
      </p>
      {(config.profiles || []).map((p) => (
        <Card key={p.id}>
          <b>{p.id}</b>{" "}
          <span className="muted">
            {p.provider} / {p.model}
          </span>
        </Card>
      ))}
      <div className="row">
        <Button onClick={onLoad}>Refresh</Button>
      </div>
    </Card>
  );
}
