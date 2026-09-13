"use client";

import { useState } from "react";
import { DATASET, BLOCK_SIZE, tokenDisplay, CORPUS } from "@/lib/corpus";
import { Vis, TokenChip } from "@/components/ui";

export default function PredictionTask() {
  const [i, setI] = useState(0);
  const ex = DATASET[i];
  // Reconstruct the raw text around the window for context
  const start = Math.min(i, CORPUS.length - BLOCK_SIZE - 1);

  return (
    <Vis
      title="The single task every LLM trains on"
      caption={
        <>
          Slide through the {DATASET.length} windows of our corpus. In each one the model reads{" "}
          {BLOCK_SIZE} tokens (the <strong>context</strong>) and must predict the <strong>next</strong>{" "}
          token. That&apos;s the whole job — repeated millions of times.
        </>
      }
    >
      {/* Input context */}
      <div className="row" style={{ gap: 5, alignItems: "center" }}>
        <span className="faint small" style={{ minWidth: 92 }}>
          context →
        </span>
        {ex.input.map((id, k) => (
          <TokenChip key={k} id={id} />
        ))}
        <span style={{ fontSize: "1.4rem", color: "var(--text-3)" }}>→</span>
        <span className="small faint" style={{ marginRight: -2 }}>
          guess
        </span>
        <span
          className="chip"
          style={{ background: "var(--accent)", borderColor: "var(--accent)", color: "#fff" }}
        >
          {tokenDisplay(ex.target[ex.target.length - 1])}
        </span>
      </div>

      <div className="vis-controls">
        <label style={{ flex: 1 }}>
          window {i + 1} / {DATASET.length}
          <input
            type="range"
            min={0}
            max={DATASET.length - 1}
            value={i}
            onChange={(e) => setI(Number(e.target.value))}
            style={{ flex: 1, width: "100%" }}
          />
        </label>
      </div>

      <div
        className="mono small"
        style={{
          marginTop: 10,
          padding: "10px 12px",
          background: "var(--surface-2)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          whiteSpace: "pre-wrap",
          lineHeight: 1.9,
        }}
      >
        {CORPUS.split("").map((c, k) => {
          const inWin = k >= start && k < start + BLOCK_SIZE;
          const isTarget = k === start + BLOCK_SIZE;
          return (
            <span
              key={k}
              style={{
                background: isTarget
                  ? "var(--accent)"
                  : inWin
                    ? "var(--accent-soft)"
                    : "transparent",
                color: isTarget ? "#fff" : inWin ? "var(--accent-strong)" : "inherit",
                borderRadius: 3,
              }}
            >
              {c === "\n" ? "⏎\n" : c === " " ? "␣" : c}
            </span>
          );
        })}
      </div>
      <div className="vis-controls">
        <span className="small faint">
          <span style={{ color: "var(--accent-strong)" }}>■ context</span> ·{" "}
          <span style={{ color: "#fff", background: "var(--accent)", padding: "0 4px", borderRadius: 3 }}>
            next token
          </span>
        </span>
      </div>
    </Vis>
  );
}
