"use client";

import { useState } from "react";
import { Vis } from "@/components/ui";

const STAGES = [
  { id: "in", title: "Input embeddings", sub: "token + position vectors", detail: "Each token is now a vector that encodes its identity and its position." },
  { id: "ln1", title: "LayerNorm", sub: "stabilize the numbers", detail: "Normalizes each vector to a stable scale so the network trains smoothly." },
  { id: "attn", title: "Self-Attention", sub: "tokens exchange information", detail: "Every token looks at all previous tokens and gathers context from them. The result is added back to the input (a residual / skip connection)." },
  { id: "ln2", title: "LayerNorm", sub: "stabilize again", detail: "Another normalization before the next sub-layer." },
  { id: "mlp", title: "Feed-Forward (MLP)", sub: "think about each token", detail: "A small 2-layer network applied to each token independently. Its result is also added back (another residual)." },
  { id: "out", title: "Output vectors", sub: "context-rich, ready for the next block", detail: "The transformed vectors. A GPT stacks many of these blocks — ToyGPT uses just one." },
];

export default function TransformerDiagram() {
  const [hover, setHover] = useState<string | null>(null);
  const active = STAGES.find((s) => s.id === hover);

  return (
    <Vis
      title="One transformer block, end to end"
      caption={
        active ? (
          <>
            <strong>{active.title}</strong> — {active.detail}
          </>
        ) : (
          <>Hover a stage. Residual (skip) connections add each sub-layer&apos;s output back to its input.</>
        )
      }
    >
      <div className="stack" style={{ alignItems: "center" }}>
        {STAGES.map((s, i) => (
          <div key={s.id} style={{ display: "flex", flexDirection: "column", alignItems: "center", width: "100%", maxWidth: 360 }}>
            <div
              onMouseEnter={() => setHover(s.id)}
              onMouseLeave={() => setHover(null)}
              style={{
                width: "100%",
                textAlign: "center",
                padding: "12px 14px",
                borderRadius: 12,
                border: `1.5px solid ${hover === s.id ? "var(--accent)" : "var(--border-strong)"}`,
                background: hover === s.id ? "var(--accent-soft)" : "var(--surface-2)",
                transition: "all .12s",
                cursor: "default",
              }}
            >
              <div style={{ fontWeight: 700, fontSize: "0.98rem" }}>{s.title}</div>
              <div className="small faint">{s.sub}</div>
              {(s.id === "attn" || s.id === "mlp") && (
                <div className="small" style={{ color: "var(--accent-strong)", marginTop: 2 }}>
                  + residual
                </div>
              )}
            </div>
            {i < STAGES.length - 1 && (
              <div style={{ color: "var(--text-3)", fontSize: "1.2rem", lineHeight: 1, padding: "3px 0" }}>
                ↓
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="callout callout-key" style={{ marginTop: 14 }}>
        <p className="callout-title">🔑 The two big ideas here</p>
        <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
          <li>
            <strong>Residual connections</strong> let information flow straight through, so stacking
            many blocks doesn&apos;t destroy the signal.
          </li>
          <li>
            <strong>Attention mixes between tokens</strong>; the <strong>MLP thinks within each
            token</strong>. Repeat, and the vectors get smarter every block.
          </li>
        </ul>
      </div>
    </Vis>
  );
}
