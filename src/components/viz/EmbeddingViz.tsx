"use client";

import { useMemo, useState } from "react";
import { VOCAB, EMBED_DIM, TOY_CONFIG } from "@/lib/corpus";
import { initParams } from "@/lib/model";
import { Vis, tokenColor } from "@/components/ui";

function heatColor(v: number, maxAbs: number): string {
  const a = Math.min(1, Math.abs(v) / maxAbs);
  if (v >= 0) return `rgba(220, 38, 38, ${a * 0.85})`;
  return `rgba(37, 99, 235, ${a * 0.85})`;
}

export default function EmbeddingViz() {
  const wte = useMemo(() => initParams(TOY_CONFIG, 42).wte, []);
  const [sel, setSel] = useState(0);
  const maxAbs = useMemo(
    () => Math.max(...wte.flat().map((x) => Math.abs(x)), 1e-6),
    [wte],
  );

  // 2D scatter on embedding dims 0 and 1
  const W = 360;
  const H = 200;
  const P = 26;
  const xs = wte.map((r) => r[0]);
  const ys = wte.map((r) => r[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const sx = (v: number) => P + ((v - minX) / (maxX - minX || 1)) * (W - 2 * P);
  const sy = (v: number) => H - P - ((v - minY) / (maxY - minY || 1)) * (H - 2 * P);

  return (
    <Vis
      title="The embedding table — each token becomes a vector"
      caption={
        <>
          Left: ToyGPT&apos;s embedding matrix — {VOCAB.length} tokens × {EMBED_DIM} numbers (red =
          positive, blue = negative). Right: each token plotted as a point using its first two
          embedding numbers. A one-hot id is just a lookup into this table.
        </>
      }
    >
      <div className="grid-2" style={{ alignItems: "center" }}>
        {/* Heatmap */}
        <div style={{ overflowX: "auto" }}>
          <div className="row" style={{ gap: 1, alignItems: "flex-end" }}>
            <span style={{ width: 34 }} />
            {Array.from({ length: EMBED_DIM }, (_, d) => (
              <span key={d} className="mono faint small" style={{ width: 20, textAlign: "center" }}>
                {d}
              </span>
            ))}
          </div>
          {wte.map((row, i) => (
            <div
              key={i}
              className="row"
              style={{ gap: 1, alignItems: "center", cursor: "pointer" }}
              onClick={() => setSel(i)}
            >
              <span
                className="mono"
                style={{
                  width: 34,
                  fontWeight: 700,
                  color: tokenColor(i),
                  background: sel === i ? "var(--surface-2)" : "transparent",
                  borderRadius: 4,
                }}
              >
                {VOCAB[i].display}
              </span>
              {row.map((v, j) => (
                <span
                  key={j}
                  style={{
                    width: 20,
                    height: 20,
                    background: heatColor(v, maxAbs),
                    borderRadius: 3,
                  }}
                  title={`${VOCAB[i].display}: dim ${j} = ${v.toFixed(3)}`}
                />
              ))}
            </div>
          ))}
        </div>

        {/* 2D scatter */}
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }}>
          <line x1={P} y1={H - P} x2={W - P} y2={H - P} stroke="var(--border-strong)" />
          <line x1={P} y1={P} x2={P} y2={H - P} stroke="var(--border-strong)" />
          {wte.map((row, i) => (
            <g key={i} onClick={() => setSel(i)} style={{ cursor: "pointer" }}>
              <circle
                cx={sx(row[0])}
                cy={sy(row[1])}
                r={sel === i ? 7 : 5}
                fill={tokenColor(i)}
                opacity={sel === i ? 1 : 0.75}
                stroke={sel === i ? "var(--text)" : "none"}
              />
              <text
                x={sx(row[0])}
                y={sy(row[1]) - 9}
                textAnchor="middle"
                fontSize={11}
                fontFamily="var(--font-mono)"
                fill={tokenColor(i)}
                fontWeight={sel === i ? 700 : 400}
              >
                {VOCAB[i].display}
              </text>
            </g>
          ))}
        </svg>
      </div>

      <div className="vis-controls">
        <span className="small muted">
          Selected token <span className="mono" style={{ color: tokenColor(sel) }}>
            {VOCAB[sel].display}
          </span>{" "}
          → vector{" "}
          <span className="mono">
            [{wte[sel].map((v) => v.toFixed(2)).join(", ")}]
          </span>
        </span>
      </div>
    </Vis>
  );
}
