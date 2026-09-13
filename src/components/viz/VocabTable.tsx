"use client";

import { useState } from "react";
import { VOCAB, VOCAB_SIZE } from "@/lib/corpus";
import { Vis, tokenColor } from "@/components/ui";

export default function VocabTable() {
  const [sel, setSel] = useState(3);

  return (
    <Vis
      title="The vocabulary — every token gets an integer id"
      caption={
        <>
          Click any row. Each token is turned into a <strong>one-hot</strong> vector: a list of{" "}
          {VOCAB_SIZE} numbers that are all <code>0</code> except a single <code>1</code> at that
          token&apos;s id. This is the model&apos;s raw input.
        </>
      }
    >
      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 4 }}>
        {VOCAB.map((t) => {
          const active = sel === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setSel(t.id)}
              style={{
                display: "grid",
                gridTemplateColumns: "36px 56px 1fr",
                gap: 12,
                alignItems: "center",
                textAlign: "left",
                padding: "5px 10px",
                borderRadius: 8,
                border: `1px solid ${active ? tokenColor(t.id) : "var(--border)"}`,
                background: active ? "var(--surface-2)" : "var(--surface)",
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              <span className="mono" style={{ fontWeight: 700, color: tokenColor(t.id) }}>
                {t.display}
              </span>
              <span className="mono small muted">id {t.id}</span>
              <span className="row" style={{ gap: 2, flexWrap: "nowrap" }}>
                {Array.from({ length: VOCAB_SIZE }, (_, j) => (
                  <span
                    key={j}
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: 3,
                      background: j === t.id ? tokenColor(t.id) : "var(--surface-3)",
                      border: `1px solid ${j === t.id ? tokenColor(t.id) : "var(--border)"}`,
                      opacity: j === t.id ? 1 : 0.6,
                    }}
                    title={`position ${j}`}
                  />
                ))}
              </span>
            </button>
          );
        })}
      </div>
      <div className="vis-controls" style={{ marginTop: 12 }}>
        <span className="small muted">
          Selected: <span className="mono">{VOCAB[sel].display}</span> → one-hot with a{" "}
          <code>1</code> at index <span className="mono">{sel}</span>, <code>0</code> everywhere
          else.
        </span>
      </div>
    </Vis>
  );
}
