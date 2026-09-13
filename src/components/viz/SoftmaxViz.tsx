"use client";

import { useMemo, useState } from "react";
import { VOCAB, tokenize, TOY_CONFIG } from "@/lib/corpus";
import { initParams, forward, softmaxRow } from "@/lib/model";
import { Vis, tokenColor } from "@/components/ui";

const CTX = tokenize("the cat "); // 8 tokens; next real token is 's' (id 6)

export default function SoftmaxViz() {
  const [temp, setTemp] = useState(1);
  const logits = useMemo(() => {
    const c = forward(initParams(TOY_CONFIG, 42), CTX);
    return c.logits[CTX.length - 1];
  }, []);
  const probs = useMemo(() => softmaxRow(logits.map((l) => l / temp)), [logits, temp]);

  const maxAbsLogit = Math.max(...logits.map((l) => Math.abs(l))) || 1;
  const maxProb = Math.max(...probs);

  return (
    <Vis
      title="From raw scores to probabilities"
      caption={
        <>
          After context <span className="mono">the cat ␣</span>, the model outputs one score per
          token (the <strong>logits</strong>). Softmax turns them into probabilities that sum to 1.
          Drag temperature to sharpen or flatten the distribution. The correct next token is{" "}
          <span className="mono">s</span>.
        </>
      }
    >
      <div className="grid-2">
        {/* Logits */}
        <div>
          <div className="card-title">Logits (raw scores)</div>
          <div className="stack" style={{ gap: 2 }}>
            {logits.map((v, i) => {
              const w = Math.abs(v) / maxAbsLogit;
              return (
                <div key={i} className="row" style={{ gap: 6, alignItems: "center" }}>
                  <span className="mono small" style={{ width: 18, color: tokenColor(i), fontWeight: 700 }}>
                    {VOCAB[i].display}
                  </span>
                  <span style={{ flex: 1, height: 14, position: "relative", background: "var(--surface-3)", borderRadius: 3 }}>
                    <span
                      style={{
                        position: "absolute",
                        top: 0,
                        height: "100%",
                        borderRadius: 3,
                        background: v >= 0 ? "var(--red)" : "var(--blue)",
                        opacity: 0.75,
                        width: `${w * 50}%`,
                        [v >= 0 ? "left" : "right"]: "50%",
                      }}
                    />
                  </span>
                  <span className="mono small" style={{ width: 44, textAlign: "right" }}>
                    {v.toFixed(1)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Probs */}
        <div>
          <div className="card-title">Probabilities (softmax)</div>
          <div className="stack" style={{ gap: 2 }}>
            {probs.map((p, i) => (
              <div key={i} className="row" style={{ gap: 6, alignItems: "center" }}>
                <span className="mono small" style={{ width: 18, color: tokenColor(i), fontWeight: 700 }}>
                  {VOCAB[i].display}
                </span>
                <span style={{ flex: 1, height: 14, background: "var(--surface-3)", borderRadius: 3, overflow: "hidden" }}>
                  <span
                    style={{
                      display: "block",
                      height: "100%",
                      background: i === 6 ? "var(--accent)" : "var(--accent)",
                      opacity: i === 6 ? 1 : 0.5,
                      width: `${(p / maxProb) * 100}%`,
                      borderRadius: 3,
                    }}
                  />
                </span>
                <span className="mono small" style={{ width: 44, textAlign: "right" }}>
                  {(p * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="vis-controls">
        <label style={{ flex: 1 }}>
          temperature = <span className="mono">{temp.toFixed(1)}</span>
          <input
            type="range"
            min={0.2}
            max={3}
            step={0.1}
            value={temp}
            onChange={(e) => setTemp(Number(e.target.value))}
            style={{ width: "100%" }}
          />
        </label>
      </div>
      <div className="vis-controls">
        <span className="small muted">
          Correct token <span className="mono">s</span> gets{" "}
          <strong>{((probs[6] ?? 0) * 100).toFixed(1)}%</strong> at temperature {temp.toFixed(1)}.
          The model&apos;s loss on this prediction is{" "}
          <span className="mono">-log({(probs[6] ?? 0).toFixed(3)}) = {(-Math.log(Math.max(probs[6], 1e-9))).toFixed(2)}</span>.
        </span>
      </div>
    </Vis>
  );
}
