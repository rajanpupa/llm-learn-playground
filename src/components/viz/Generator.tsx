"use client";

import { useMemo, useState } from "react";
import { initParams, mulberry32, generate } from "@/lib/model";
import { initAdam, trainStep } from "@/lib/train";
import { buildDataset, tokenize, tokenDisplay, TOY_CONFIG } from "@/lib/corpus";
import { Vis, tokenColor } from "@/components/ui";

export default function Generator() {
  // Train a model once (synchronously, ~tens of ms) so we can sample from it.
  const model = useMemo(() => {
    const p = initParams(TOY_CONFIG, 42);
    const adam = initAdam(TOY_CONFIG);
    const ds = buildDataset();
    for (let i = 0; i < 500; i++) {
      const s = (i * 4) % ds.length;
      const batch = Array.from({ length: 4 }, (_, k) => ds[(s + k) % ds.length]);
      trainStep(p, adam, batch);
    }
    return p;
  }, []);

  const [temp, setTemp] = useState(0.7);
  const [topK, setTopK] = useState(0);
  const [seed, setSeed] = useState(1);
  const [out, setOut] = useState<number[]>(() =>
    generate(model, tokenize("the"), 42, 0.7, 0, mulberry32(1)),
  );

  const run = () => {
    setOut(generate(model, tokenize("the"), 42, temp, topK, mulberry32(seed)));
    setSeed((s) => s + 1);
  };

  return (
    <Vis
      title="Sample new text from the trained model"
      caption={
        <>
          The model was just trained (in-browser, 500 steps) on our corpus. Generation is{" "}
          <em>autoregressive</em>: it predicts one token, appends it, and feeds everything back to
          predict the next. Drag <strong>temperature</strong> to control randomness, and{" "}
          <strong>top-k</strong> to restrict to the k most likely tokens.
        </>
      }
    >
      <div className="vis-controls">
        <button className="btn btn-primary" onClick={run}>
          Generate
        </button>
        <label>
          temp = <span className="mono">{temp.toFixed(1)}</span>
          <input type="range" min={0.1} max={2} step={0.1} value={temp} onChange={(e) => setTemp(Number(e.target.value))} />
        </label>
        <label>
          top-k = <span className="mono">{topK === 0 ? "off" : topK}</span>
          <input type="range" min={0} max={13} step={1} value={topK} onChange={(e) => setTopK(Number(e.target.value))} />
        </label>
      </div>

      <div
        style={{
          marginTop: 14,
          padding: 16,
          background: "var(--surface-2)",
          border: "1px solid var(--border)",
          borderRadius: 12,
          fontFamily: "var(--font-mono)",
          fontSize: "1.05rem",
          lineHeight: 2,
          whiteSpace: "pre-wrap",
          wordBreak: "break-all",
        }}
      >
        {out.map((id, i) =>
          id === 7 ? (
            <span key={i} style={{ color: "var(--text-3)" }}>
              ⏎{"\n"}
            </span>
          ) : (
            <span key={i} style={{ color: tokenColor(id) }}>
              {tokenDisplay(id)}
            </span>
          ),
        )}
      </div>

      <div className="vis-controls">
        <span className="small muted">
          <span style={{ color: "var(--text-2)" }}>prompt:</span> <span className="mono">the</span>{" "}
          <span style={{ color: "var(--text-3)" }}>→</span> generated{" "}
          <strong>{out.length - 3}</strong> new tokens (colored by token).
        </span>
      </div>
    </Vis>
  );
}
