"use client";

import { useEffect, useState } from "react";
import { Vis } from "@/components/ui";

const NODES = [
  { id: "emb", label: "Embeddings", fwd: "tokens → vectors", bwd: "updates every embedding weight" },
  { id: "attn", label: "Self-Attention", fwd: "tokens exchange context", bwd: "updates Q/K/V & output weights" },
  { id: "mlp", label: "MLP", fwd: "per-token computation", bwd: "updates both MLP layers" },
  { id: "logits", label: "Logits", fwd: "raw scores per token", bwd: "receives the loss gradient" },
  { id: "prob", label: "Softmax → probabilities", fwd: "scores → probabilities", bwd: "back-propagates the error" },
  { id: "loss", label: "Loss", fwd: "compares to the true token", bwd: "the starting point of every gradient" },
];

type Step = { node: number; dir: "fwd" | "bwd"; text: string };

const STEPS: Step[] = [
  ...NODES.map((n, i) => ({ node: i, dir: "fwd" as const, text: n.fwd })),
  ...[...NODES].reverse().map((n, ri) => ({ node: NODES.length - 1 - ri, dir: "bwd" as const, text: n.bwd })),
];

export default function BackpropViz() {
  const [idx, setIdx] = useState(-1);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    if (idx >= STEPS.length - 1) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setIdx((i) => i + 1), 620);
    return () => clearTimeout(t);
  }, [playing, idx]);

  const active = idx >= 0 ? STEPS[idx] : null;

  return (
    <Vis
      title="Two passes: forward to measure, backward to fix"
      caption={
        <>
          Press <strong>Play</strong>. The <span style={{ color: "var(--green)" }}>forward pass</span>{" "}
          computes a prediction and a loss; then the{" "}
          <span style={{ color: "var(--red)" }}>backward pass</span> walks the same path in reverse,
          using the chain rule to compute how much each weight contributed to the error.
        </>
      }
    >
      <div className="vis-controls">
        <button className="btn btn-primary btn-sm" onClick={() => { setPlaying(true); setIdx(-1); }}>
          Play
        </button>
        <button className="btn btn-ghost btn-sm" onClick={() => { setPlaying(false); setIdx(-1); }}>
          Stop
        </button>
      </div>

      <div className="stack" style={{ alignItems: "center", marginTop: 8 }}>
        {NODES.map((n, i) => {
          const isActive = active?.node === i;
          const dir = isActive ? active!.dir : null;
          const border = dir === "fwd" ? "var(--green)" : dir === "bwd" ? "var(--red)" : "var(--border-strong)";
          const bg = dir === "fwd" ? "color-mix(in srgb, var(--green) 10%, var(--surface))" : dir === "bwd" ? "color-mix(in srgb, var(--red) 10%, var(--surface))" : "var(--surface-2)";
          return (
            <div key={n.id} style={{ width: "100%", maxWidth: 360, display: "flex", flexDirection: "column", alignItems: "center" }}>
              <div
                style={{
                  width: "100%",
                  textAlign: "center",
                  padding: "10px 14px",
                  borderRadius: 10,
                  border: `1.5px solid ${border}`,
                  background: bg,
                  transition: "all .12s",
                }}
              >
                <div style={{ fontWeight: 700, fontSize: "0.95rem" }}>
                  {n.label}
                  {isActive && (dir === "fwd" ? " ↓" : " ↑")}
                </div>
              </div>
              {i < NODES.length - 1 && (
                <div style={{ color: "var(--text-3)", lineHeight: 1, padding: "3px 0" }}>
                  {dir === "bwd" && isActive ? "↑" : "↓"}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="vis-controls" style={{ justifyContent: "center" }}>
        {active ? (
          <span className="small" style={{ color: active.dir === "fwd" ? "var(--green)" : "var(--red)" }}>
            <strong>{active.dir === "fwd" ? "Forward" : "Backward"}:</strong> {active.text}
          </span>
        ) : (
          <span className="small faint">Ready — press Play.</span>
        )}
      </div>
    </Vis>
  );
}
