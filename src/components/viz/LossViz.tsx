"use client";

import { useMemo, useState } from "react";
import { VOCAB, tokenize, tokenDisplay, TOY_CONFIG } from "@/lib/corpus";
import { initParams, forward } from "@/lib/model";
import { Vis, tokenColor } from "@/components/ui";

const CTX = tokenize("the cat "); // 8 tokens
const TARGETS = tokenize("the cat sat").slice(1, 9);

export default function LossViz() {
  const [pos, setPos] = useState(7);
  const { probs } = useMemo(() => {
    const c = forward(initParams(TOY_CONFIG, 42), CTX);
    return { probs: c.probs };
  }, []);

  const p = probs[pos];
  const target = TARGETS[pos];
  const lossAt = -Math.log(Math.max(p[target], 1e-9));
  const meanLoss = useMemo(() => {
    let s = 0;
    for (let i = 0; i < probs.length; i++) s += -Math.log(Math.max(probs[i][TARGETS[i]], 1e-9));
    return s / probs.length;
  }, [probs]);
  const maxP = Math.max(...p);

  return (
    <Vis
      title="Cross-entropy — how wrong is the model?"
      caption={
        <>
          At each position, compare the model&apos;s predicted distribution to the one true next
          token. The loss is <span className="mono">-log p(correct)</span>: confident &amp; right ≈ 0,
          confident &amp; wrong ≈ huge. Average over all positions = the training loss.
        </>
      }
    >
      {/* context with position */}
      <div className="row" style={{ gap: 5, alignItems: "center", marginBottom: 12 }}>
        <span className="faint small" style={{ minWidth: 92 }}>
          context
        </span>
        {CTX.map((id, i) => (
          <button
            key={i}
            onClick={() => setPos(i)}
            className="chip"
            style={{
              background: i === pos ? "var(--accent-soft)" : "var(--surface-2)",
              borderColor: i === pos ? "var(--accent)" : "var(--border)",
              cursor: "pointer",
            }}
          >
            {tokenDisplay(id)}
          </button>
        ))}
        <span style={{ color: "var(--text-3)" }}>→</span>
        <span className="chip" style={{ background: "var(--accent)", borderColor: "var(--accent)", color: "#fff" }}>
          {tokenDisplay(target)}
        </span>
      </div>

      {/* predicted distribution */}
      <div className="stack" style={{ gap: 2 }}>
        {p.map((v, i) => (
          <div key={i} className="row" style={{ gap: 6, alignItems: "center" }}>
            <span className="mono small" style={{ width: 18, color: tokenColor(i), fontWeight: 700 }}>
              {VOCAB[i].display}
            </span>
            <span style={{ flex: 1, height: 14, background: "var(--surface-3)", borderRadius: 3, overflow: "hidden" }}>
              <span
                style={{
                  display: "block",
                  height: "100%",
                  background: i === target ? "var(--accent)" : "var(--accent)",
                  opacity: i === target ? 1 : 0.4,
                  width: `${(v / maxP) * 100}%`,
                  borderRadius: 3,
                }}
              />
            </span>
            <span className="mono small" style={{ width: 44, textAlign: "right" }}>
              {(v * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>

      <div className="row" style={{ marginTop: 12, gap: 18, flexWrap: "wrap" }}>
        <span className="small muted">
          p(correct) = <span className="mono">{p[target].toFixed(3)}</span> → loss{" "}
          <span className="mono">-log p = <strong>{lossAt.toFixed(2)}</strong></span>
        </span>
        <span className="small muted">
          mean loss over this window: <span className="mono"><strong>{meanLoss.toFixed(2)}</strong></span>
        </span>
      </div>
    </Vis>
  );
}
