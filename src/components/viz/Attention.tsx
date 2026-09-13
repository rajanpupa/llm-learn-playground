"use client";

import { useMemo, useState } from "react";
import { initParams, forward } from "@/lib/model";
import { tokenize, tokenDisplay, TOY_CONFIG } from "@/lib/corpus";
import { Vis } from "@/components/ui";

const SAMPLE = "the cat ";
const SAMPLE_IDS = tokenize(SAMPLE);

function useHeads() {
  return useMemo(() => {
    const cache = forward(initParams(TOY_CONFIG, 42), SAMPLE_IDS);
    return {
      labels: SAMPLE_IDS.map((id) => tokenDisplay(id)),
      heads: cache.attn.probs, // [H][T][T]
    };
  }, []);
}

export function AttentionHeatmap({
  probs,
  labels,
  title,
  caption,
}: {
  probs: number[][];
  labels: string[];
  title: string;
  caption?: React.ReactNode;
}) {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const T = probs.length;
  const M = 34; // margin for labels
  const cell = 36;
  const size = M + T * cell;

  return (
    <Vis title={title} caption={caption}>
      <svg viewBox={`0 0 ${size} ${size}`} style={{ width: "100%", maxWidth: 400 }}>
        {/* column (key) labels */}
        {labels.map((l, c) => (
          <text
            key={c}
            x={M + c * cell + cell / 2}
            y={M - 8}
            textAnchor="middle"
            fontSize={13}
            fontFamily="var(--font-mono)"
            fill={hover?.c === c ? "var(--accent)" : "var(--text-3)"}
            fontWeight={hover?.c === c ? 700 : 400}
          >
            {l}
          </text>
        ))}
        {/* row (query) labels */}
        {labels.map((l, r) => (
          <text
            key={r}
            x={M - 8}
            y={M + r * cell + cell / 2 + 4}
            textAnchor="middle"
            fontSize={13}
            fontFamily="var(--font-mono)"
            fill={hover?.r === r ? "var(--accent)" : "var(--text-3)"}
            fontWeight={hover?.r === r ? 700 : 400}
          >
            {l}
          </text>
        ))}
        {/* cells */}
        {probs.map((row, r) =>
          row.map((v, c) => {
            const rowMax = Math.max(...row) || 1;
            const masked = c > r;
            const isHover = hover && (hover.r === r || hover.c === c);
            return (
              <rect
                key={`${r}-${c}`}
                x={M + c * cell}
                y={M + r * cell}
                width={cell - 2}
                height={cell - 2}
                rx={4}
                fill={masked ? "var(--surface-3)" : "var(--accent)"}
                fillOpacity={masked ? 1 : 0.06 + 0.94 * (v / rowMax)}
                stroke={isHover ? "var(--text)" : "none"}
                strokeWidth={isHover ? 1.5 : 0}
                onMouseEnter={() => setHover({ r, c })}
                onMouseLeave={() => setHover(null)}
                style={{ cursor: "pointer" }}
              >
                <title>{`query ${labels[r]} → key ${labels[c]}: ${(v * 100).toFixed(1)}%`}</title>
              </rect>
            );
          }),
        )}
        <text
          x={M + T * cell - 4}
          y={M + T * cell + 16}
          textAnchor="end"
          fontSize={11}
          fill="var(--text-3)"
          fontFamily="var(--font-sans)"
        >
          row = query (looking) · column = key (looked at)
        </text>
      </svg>
      <div className="vis-controls">
        {hover ? (
          <span className="small muted">
            Token <span className="mono">{labels[hover.r]}</span> gives{" "}
            <strong>
              {(probs[hover.r][hover.c] * 100).toFixed(1)}%
            </strong>{" "}
            of its attention to <span className="mono">{labels[hover.c]}</span>
            {hover.c > hover.r ? " (masked — the future is hidden)" : ""}.
          </span>
        ) : (
          <span className="small faint">Hover a square to inspect a weight.</span>
        )}
      </div>
    </Vis>
  );
}

export default function AttentionViz() {
  const { heads, labels } = useHeads();
  return (
    <AttentionHeatmap
      probs={heads[0]}
      labels={labels}
      title="Attention weights (head 0) for “the cat ␣”"
      caption={
        <>
          Each row is a token deciding how much to look at each <em>previous</em> token (the{" "}
          <strong>causal mask</strong> hides the future — everything above the diagonal is blocked).
          Rows are softmax-ed, so each row sums to 100%. These are real weights from the untrained
          model: roughly even, because it hasn&apos;t learned yet what matters.
        </>
      }
    />
  );
}

export function MultiHeadViz() {
  const { heads, labels } = useHeads();
  return (
    <Vis
      title="Two heads in parallel"
      caption={
        <>
          ToyGPT runs <strong>2 attention heads</strong> at once on the same input, each producing
          its own attention pattern, then concatenates the results. With more data, different heads
          learn different jobs — one might track grammar, another the previous word.
        </>
      }
    >
      <div className="grid-2">
        {heads.map((h, i) => (
          <AttentionHeatmap
            key={i}
            probs={h}
            labels={labels}
            title={`Head ${i}`}
            caption={i === 0 ? "e.g. tends to look at the previous token" : "e.g. tends to look further back"}
          />
        ))}
      </div>
    </Vis>
  );
}
