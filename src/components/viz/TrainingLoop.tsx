"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { initParams, mulberry32, generate } from "@/lib/model";
import { initAdam, trainStep, evalLoss } from "@/lib/train";
import { buildDataset, tokenDisplay, BLOCK_SIZE, TOY_CONFIG } from "@/lib/corpus";
import { Vis } from "@/components/ui";

const BATCH = 4;
const SEED = 7;

function sample(p: ReturnType<typeof initParams>): string {
  return generate(p, [0], 32, 0.7, 0, mulberry32(SEED))
    .map((id) => tokenDisplay(id))
    .join("");
}

export default function TrainingLoop() {
  const ds = useMemo(() => buildDataset(), []);
  const paramsRef = useRef(initParams(TOY_CONFIG, 42));
  const adamRef = useRef(initAdam(TOY_CONFIG));
  const stepRef = useRef(0);

  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [step, setStep] = useState(0);
  const [loss, setLoss] = useState(() => evalLoss(paramsRef.current, ds));
  const [history, setHistory] = useState<number[]>([evalLoss(paramsRef.current, ds)]);
  const [before, setBefore] = useState(() => sample(paramsRef.current));
  const [current, setCurrent] = useState(() => sample(paramsRef.current));

  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const loop = () => {
      for (let k = 0; k < speed; k++) {
        const s = (stepRef.current * BATCH) % ds.length;
        const batch = Array.from({ length: BATCH }, (_, j) => ds[(s + j) % ds.length]);
        trainStep(paramsRef.current, adamRef.current, batch);
        stepRef.current += 1;
      }
      const l = evalLoss(paramsRef.current, ds);
      setStep(stepRef.current);
      setLoss(l);
      setHistory((h) => {
        const next = h.length > 1200 ? h.slice(-800) : h;
        return [...next, l];
      });
      if (stepRef.current % 24 < speed) setCurrent(sample(paramsRef.current));
      if (l < 0.03 || stepRef.current > 6000) {
        setRunning(false);
        return;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, speed, ds]);

  const reset = () => {
    setRunning(false);
    paramsRef.current = initParams(TOY_CONFIG, 42);
    adamRef.current = initAdam(TOY_CONFIG);
    stepRef.current = 0;
    setStep(0);
    setLoss(evalLoss(paramsRef.current, ds));
    setHistory([evalLoss(paramsRef.current, ds)]);
    setBefore(sample(paramsRef.current));
    setCurrent(sample(paramsRef.current));
  };

  // Loss curve
  const W = 620;
  const H = 220;
  const maxL = 2.8;
  const n = Math.max(history.length, 2);
  const pts = history
    .map((l, i) => `${(i / (n - 1)) * W},${H - 8 - (Math.min(l, maxL) / maxL) * (H - 30)}`)
    .join(" ");

  return (
    <Vis
      title="Watch ToyGPT actually train — live, in your browser"
      caption={
        <>
          This is a real {BLOCK_SIZE}-token transformer with real backprop and the Adam optimizer,
          training on our 4-line corpus. Press <strong>Train</strong> and watch the loss fall as the
          generated text goes from gibberish to something that looks like English.
        </>
      }
    >
      <div className="vis-controls">
        <button className="btn btn-primary" onClick={() => setRunning((r) => !r)}>
          {running ? "Pause" : "Train"}
        </button>
        <button className="btn btn-ghost" onClick={reset} disabled={running}>
          Reset
        </button>
        <span className="small muted">
          step <strong>{step}</strong> · loss <strong>{loss.toFixed(3)}</strong>
        </span>
        <span className="row" style={{ gap: 6 }}>
          {[1, 4, 16].map((s) => (
            <button
              key={s}
              className="btn btn-ghost btn-sm"
              style={speed === s ? { borderColor: "var(--accent)", color: "var(--accent)" } : {}}
              onClick={() => setSpeed(s)}
            >
              {s}×
            </button>
          ))}
        </span>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", marginTop: 8 }}>
        <line x1={0} y1={H - 8 - (2.565 / maxL) * (H - 30)} x2={W} y2={H - 8 - (2.565 / maxL) * (H - 30)} stroke="var(--border-strong)" strokeDasharray="4 4" />
        <text x={4} y={H - 8 - (2.565 / maxL) * (H - 30) - 4} fontSize={11} fill="var(--text-3)" fontFamily="var(--font-mono)">
          random guess (ln 13 ≈ 2.57)
        </text>
        <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
        <line x1={0} y1={H - 8} x2={W} y2={H - 8} stroke="var(--border-strong)" />
        <text x={4} y={H - 14} fontSize={11} fill="var(--text-3)" fontFamily="var(--font-mono)">
          loss → (lower is better)
        </text>
      </svg>

      <div className="grid-2" style={{ marginTop: 12 }}>
        <div className="card card-pad">
          <div className="card-title">Before training</div>
          <div className="mono" style={{ whiteSpace: "pre-wrap", fontSize: "0.9rem", color: "var(--text-3)" }}>
            {before}
          </div>
        </div>
        <div className="card card-pad">
          <div className="card-title">Now</div>
          <div className="mono" style={{ whiteSpace: "pre-wrap", fontSize: "0.9rem", color: "var(--text)" }}>
            {current}
          </div>
        </div>
      </div>
    </Vis>
  );
}
