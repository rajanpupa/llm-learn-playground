"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Vis } from "@/components/ui";

// A simple bowl-shaped loss over two weights: L = (w1-1)^2 + 3*(w2+0.5)^2
const MIN = [1, -0.5] as const;
const X0 = -2;
const X1 = 4;
const Y0 = -2.5;
const Y1 = 1.5;
const L = (w1: number, w2: number) => (w1 - 1) ** 2 + 3 * (w2 + 0.5) ** 2;
const grad = (w1: number, w2: number): [number, number] => [2 * (w1 - 1), 6 * (w2 + 0.5)];

const W = 460;
const H = 300;
const NX = 46;
const NY = 30;

export default function LossLandscape() {
  const svgRef = useRef<SVGSVGElement>(null);
  const [path, setPath] = useState<[number, number][]>([]);
  const [running, setRunning] = useState(false);
  const [lr, setLr] = useState(0.1);
  const curRef = useRef<[number, number]>([3.2, 1.0]);
  const countRef = useRef(0);

  const px = (v: number) => ((v - X0) / (X1 - X0)) * W;
  const py = (v: number) => H - ((v - Y0) / (Y1 - Y0)) * H;

  const cells = useMemo(() => {
    const out: { x: number; y: number; v: number }[] = [];
    const cw = W / NX;
    const ch = H / NY;
    for (let i = 0; i < NX; i++)
      for (let j = 0; j < NY; j++) {
        const w1 = X0 + (i + 0.5) / NX * (X1 - X0);
        const w2 = Y0 + (j + 0.5) / NY * (Y1 - Y0);
        out.push({ x: i * cw, y: j * ch, v: L(w1, w2) });
      }
    return out;
  }, []);

  const step = () => {
    const [w1, w2] = curRef.current;
    const [g1, g2] = grad(w1, w2);
    const nw1 = w1 - lr * g1;
    const nw2 = w2 - lr * g2;
    curRef.current = [nw1, nw2];
    setPath((prev) => [...prev, [nw1, nw2]]);
    countRef.current += 1;
    if (Math.hypot(g1, g2) < 1e-3 || countRef.current > 160 || Math.hypot(nw1, nw2) > 1e4) {
      setRunning(false);
    }
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(step, 55);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, lr]);

  const start = (p: [number, number]) => {
    setPath([p]);
    curRef.current = p;
    countRef.current = 0;
  };

  const onClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const vx = ((e.clientX - rect.left) / rect.width) * W;
    const vy = ((e.clientY - rect.top) / rect.height) * H;
    const w1 = X0 + (vx / W) * (X1 - X0);
    const w2 = Y1 - (vy / H) * (Y1 - Y0);
    start([w1, w2]);
    setRunning(false);
  };

  const cur = path.length > 0 ? path[path.length - 1] : null;
  const maxL = L(X0, Y0);

  return (
    <Vis
      title="The loss landscape — training walks downhill"
      caption={
        <>
          Two weights, one loss (the height of the surface). Click anywhere to drop a starting
          point, then <strong>Run</strong>. Each step moves opposite the gradient (the steepest
          downhill direction). Try raising the learning rate to watch it overshoot.
        </>
      }
    >
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} onClick={onClick} style={{ width: "100%", cursor: "crosshair" }}>
        {cells.map((c, i) => (
          <rect
            key={i}
            x={c.x}
            y={c.y}
            width={W / NX + 0.5}
            height={H / NY + 0.5}
            fill="var(--accent)"
            fillOpacity={0.04 + 0.85 * (c.v / maxL)}
          />
        ))}
        {/* minimum */}
        <circle cx={px(MIN[0])} cy={py(MIN[1])} r={4} fill="var(--green)" />
        <text x={px(MIN[0]) + 8} y={py(MIN[1]) + 4} fontSize={11} fill="var(--green)" fontFamily="var(--font-mono)">
          minimum
        </text>
        {/* path */}
        {path.length > 1 && (
          <polyline
            points={path.map((p) => `${px(p[0])},${py(p[1])}`).join(" ")}
            fill="none"
            stroke="var(--text)"
            strokeWidth={2}
            opacity={0.85}
          />
        )}
        {path.map((p, i) => (
          <circle key={i} cx={px(p[0])} cy={py(p[1])} r={i === path.length - 1 ? 4 : 2.5} fill={i === 0 ? "var(--amber)" : "var(--text)"} />
        ))}
      </svg>

      <div className="vis-controls">
        <button className="btn btn-primary btn-sm" onClick={() => setRunning((r) => !r)} disabled={path.length === 0}>
          {running ? "Pause" : "Run"}
        </button>
        <button className="btn btn-ghost btn-sm" onClick={() => { start([3.2, 1.0]); setRunning(false); }}>
          Reset
        </button>
        <label>
          lr = <span className="mono">{lr.toFixed(2)}</span>
          <input type="range" min={0.02} max={0.5} step={0.01} value={lr} onChange={(e) => setLr(Number(e.target.value))} />
        </label>
      </div>
      <div className="vis-controls">
        <span className="small muted">
          {cur ? (
            <>
              weights = <span className="mono">({cur[0].toFixed(2)}, {cur[1].toFixed(2)})</span> · loss ={" "}
              <span className="mono">{L(cur[0], cur[1]).toFixed(3)}</span> · step {countRef.current}
            </>
          ) : (
            "Click the surface to place a starting point."
          )}
        </span>
      </div>
    </Vis>
  );
}
