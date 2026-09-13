"use client";

import { useState } from "react";
import { Vis } from "@/components/ui";

const G: [number, number] = [2.5, 1.5];
const G_NORM = Math.hypot(...G);

export default function OptimizerViz() {
  const [clip, setClip] = useState(1.0);
  const clipped = G_NORM > clip;
  const scale = 30;
  const cx = 150;
  const cy = 120;
  const gx = G[0] * scale;
  const gy = -G[1] * scale;
  const cgx = clipped ? gx * (clip / G_NORM) : gx;
  const cgy = clipped ? gy * (clip / G_NORM) : gy;

  // Learning-rate schedule: warmup then cosine decay
  const lrPts = (() => {
    const pts: string[] = [];
    for (let s = 0; s <= 100; s++) {
      let lr: number;
      if (s < 10) lr = 0.02 * (s / 10);
      else lr = 0.02 * 0.5 * (1 + Math.cos(Math.PI * ((s - 10) / 90)));
      const x = 10 + (s / 100) * 280;
      const y = 120 - (lr / 0.02) * 100;
      pts.push(`${x},${y}`);
    }
    return pts.join(" ");
  })();

  return (
    <Vis
      title="Two safety rails: gradient clipping & a learning-rate schedule"
      caption={
        <>
          <strong>Clipping</strong> caps the size of an update so one weird example can&apos;t send
          the model flying. <strong>Scheduling</strong> warms the learning rate up then cools it
          down, so training is stable at the start and precise at the end.
        </>
      }
    >
      <div className="grid-2">
        {/* Clipping */}
        <div>
          <div className="card-title">Gradient clipping</div>
          <svg viewBox="0 0 300 240" style={{ width: "100%" }}>
            <line x1={0} y1={cy} x2={300} y2={cy} stroke="var(--border-strong)" />
            <line x1={cx} y1={240} x2={cx} y2={0} stroke="var(--border-strong)" />
            <circle cx={cx} cy={cy} r={clip * scale} fill="none" stroke="var(--text-3)" strokeDasharray="4 4" />
            {/* raw gradient */}
            <line x1={cx} y1={cy} x2={cx + gx} y2={cy + gy} stroke="var(--blue)" strokeWidth={3} />
            <circle cx={cx + gx} cy={cy + gy} r={5} fill="var(--blue)" />
            {/* clipped gradient */}
            {clipped && (
              <>
                <line x1={cx} y1={cy} x2={cx + cgx} y2={cy + cgy} stroke="var(--red)" strokeWidth={3} />
                <circle cx={cx + cgx} cy={cy + cgy} r={5} fill="var(--red)" />
              </>
            )}
            <text x={cx + gx + 8} y={cy + gy - 6} fontSize={11} fill="var(--blue)" fontFamily="var(--font-mono)">
              raw
            </text>
            {clipped && (
              <text x={cx + cgx - 26} y={cy + cgy + 16} fontSize={11} fill="var(--red)" fontFamily="var(--font-mono)">
                clipped
              </text>
            )}
          </svg>
          <div className="vis-controls">
            <label style={{ width: "100%" }}>
              clip norm = <span className="mono">{clip.toFixed(1)}</span>
              <input type="range" min={0.5} max={4} step={0.1} value={clip} onChange={(e) => setClip(Number(e.target.value))} style={{ width: "100%" }} />
            </label>
          </div>
        </div>

        {/* LR schedule */}
        <div>
          <div className="card-title">Learning-rate schedule</div>
          <svg viewBox="0 0 300 140" style={{ width: "100%" }}>
            <polyline points={lrPts} fill="none" stroke="var(--accent)" strokeWidth={2.5} />
            <line x1={10} y1={120} x2={290} y2={120} stroke="var(--border-strong)" />
            <line x1={10} y1={20} x2={10} y2={120} stroke="var(--border-strong)" />
            <text x={20} y={60} fontSize={11} fill="var(--text-2)" fontFamily="var(--font-sans)">warmup</text>
            <text x={190} y={40} fontSize={11} fill="var(--text-2)" fontFamily="var(--font-sans)">cosine decay</text>
            <text x={12} y={132} fontSize={10} fill="var(--text-3)" fontFamily="var(--font-mono)">0</text>
            <text x={262} y={132} fontSize={10} fill="var(--text-3)" fontFamily="var(--font-mono)">steps →</text>
          </svg>
          <p className="small muted" style={{ marginTop: 8 }}>
            Adam also keeps a running average (momentum) of gradients so updates are smoother and
            faster to converge.
          </p>
        </div>
      </div>
    </Vis>
  );
}
