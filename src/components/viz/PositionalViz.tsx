"use client";

import { useMemo, useState } from "react";
import { EMBED_DIM, TOY_CONFIG } from "@/lib/corpus";
import { initParams } from "@/lib/model";
import { Vis } from "@/components/ui";

function heatColor(v: number, maxAbs: number): string {
  const a = Math.min(1, Math.abs(v) / maxAbs);
  if (v >= 0) return `rgba(220, 38, 38, ${a * 0.85})`;
  return `rgba(37, 99, 235, ${a * 0.85})`;
}

export default function PositionalViz() {
  const wpe = useMemo(() => initParams(TOY_CONFIG, 42).wpe, []);
  const [pos, setPos] = useState(0);
  const maxAbs = useMemo(() => Math.max(...wpe.flat().map((x) => Math.abs(x)), 1e-6), [wpe]);

  return (
    <Vis
      title="Position embeddings — a unique vector for each slot"
      caption={
        <>
          ToyGPT adds a <strong>position vector</strong> to every token&apos;s embedding, so the
          same token at a different spot looks different. Hover a position to see its vector. This
          is how a model that processes all tokens at once still knows their order.
        </>
      }
    >
      <div className="row" style={{ gap: 3, alignItems: "flex-end" }}>
        <span className="faint small" style={{ width: 60 }}>
          dims →
        </span>
        {Array.from({ length: EMBED_DIM }, (_, d) => (
          <span key={d} className="mono faint small" style={{ width: 20, textAlign: "center" }}>
            {d}
          </span>
        ))}
      </div>
      {wpe.map((row, p) => (
        <div
          key={p}
          className="row"
          style={{ gap: 3, alignItems: "center", cursor: "pointer" }}
          onMouseEnter={() => setPos(p)}
          onClick={() => setPos(p)}
        >
          <span
            className="mono"
            style={{
              width: 60,
              fontWeight: 700,
              color: pos === p ? "var(--accent)" : "var(--text-3)",
            }}
          >
            pos {p}
          </span>
          {row.map((v, d) => (
            <span
              key={d}
              style={{
                width: 20,
                height: 20,
                background: heatColor(v, maxAbs),
                borderRadius: 3,
                outline: pos === p ? "1.5px solid var(--accent)" : "none",
              }}
              title={`pos ${p}, dim ${d} = ${v.toFixed(3)}`}
            />
          ))}
        </div>
      ))}

      <div className="vis-controls">
        <span className="small muted">
          pos <span className="mono">{pos}</span> adds{" "}
          <span className="mono">
            [{wpe[pos].map((v) => v.toFixed(2)).join(", ")}]
          </span>{" "}
          to whatever token sits in slot {pos}.
        </span>
      </div>

      <div className="callout callout-tip" style={{ marginTop: 12 }}>
        <p className="callout-title">💡 Intuition: why not just use 0, 1, 2, …?</p>
        <p className="small" style={{ margin: 0 }}>
          Counting numbers grow unbounded and have no two numbers that are &ldquo;close&rdquo; in a
          useful way. A learned (or sinusoidal) vector gives every position a rich, comparable
          fingerprint across {EMBED_DIM} dimensions — which the model can blend with meaning.
        </p>
      </div>
    </Vis>
  );
}
