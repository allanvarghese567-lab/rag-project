import { Card } from "./Card";
import { Button } from "./Button";
import type { EmbeddingsConfig } from "../types";

interface EmbeddingsPanelProps {
  config: EmbeddingsConfig | null;
  hasApi: boolean;
  onLoad: () => void;
}

/** Embedding profile summary. */
export function EmbeddingsPanel({ config, hasApi, onLoad }: EmbeddingsPanelProps) {
  return (
    <Card title="Embeddings">
      <p className="muted">
        {config ? (
          <>
            Active: <code>{config.active}</code>
          </>
        ) : (
          "Connect API or load /embeddings"
        )}
      </p>
      {config?.profiles?.map((p) => (
        <Card key={p.id}>
          <b>{p.id}</b>{" "}
          <span className="muted">
            {p.provider} / {p.model}
          </span>
        </Card>
      ))}
      <div className="row">
        <Button primary disabled={!hasApi} onClick={onLoad}>
          Load from API
        </Button>
      </div>
    </Card>
  );
}
