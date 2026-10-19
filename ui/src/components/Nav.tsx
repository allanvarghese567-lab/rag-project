import type { TabId } from "../types";

const TABS: { id: TabId; label: string }[] = [
  { id: "api", label: "Live API" },
  { id: "llms", label: "LLMs" },
  { id: "emb", label: "Embeddings" },
  { id: "jobs", label: "Jobs" },
  { id: "conn", label: "Connect" },
];

interface NavProps {
  active: TabId;
  onChange: (id: TabId) => void;
}

/** Sticky tab navigation. */
export function Nav({ active, onChange }: NavProps) {
  return (
    <nav>
      {TABS.map((t) => (
        <button
          key={t.id}
          type="button"
          className={t.id === active ? "active" : ""}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </nav>
  );
}
