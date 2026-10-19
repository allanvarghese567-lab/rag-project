import { Card } from "./Card";
import { Button } from "./Button";
import { Field, TextInput } from "./Field";
import type { SessionConfig } from "../types";

interface ConnectPanelProps {
  session: SessionConfig;
  onChange: (patch: Partial<SessionConfig>) => void;
  onSaveApi: () => void;
  onConnectGithub: () => void;
}

/** API base URL + optional GitHub credentials. */
export function ConnectPanel({
  session,
  onChange,
  onSaveApi,
  onConnectGithub,
}: ConnectPanelProps) {
  return (
    <>
      <Card title="Live RAG API">
        <Field label="API base URL">
          <TextInput
            value={session.apiBase}
            placeholder="https://rag-api.onrender.com"
            onChange={(e) => onChange({ apiBase: e.target.value.trim() })}
          />
        </Field>
        <div className="row" style={{ marginTop: 8 }}>
          <Button primary onClick={onSaveApi}>
            Save API URL
          </Button>
        </div>
      </Card>

      <Card title="GitHub">
        <Field label="Owner">
          <TextInput
            value={session.owner}
            onChange={(e) => onChange({ owner: e.target.value.trim() })}
          />
        </Field>
        <Field label="Repo">
          <TextInput
            value={session.repo}
            onChange={(e) => onChange({ repo: e.target.value.trim() })}
          />
        </Field>
        <Field label="Token">
          <TextInput
            type="password"
            value={session.token}
            onChange={(e) => onChange({ token: e.target.value.trim() })}
          />
        </Field>
        <div className="row" style={{ marginTop: 10 }}>
          <Button primary onClick={onConnectGithub}>
            Connect
          </Button>
        </div>
      </Card>
    </>
  );
}
