"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Vis } from "@/components/ui";
import { formatLessonNumber } from "@/lib/lessons";

const STAGES = [
  { n: 1, slug: "text-into-tokens", title: "Text → tokens", desc: "Split text into characters" },
  { n: 2, slug: "vocabulary-token-ids", title: "Token IDs", desc: "One-hot vectors" },
  { n: 3, slug: "embeddings", title: "Embeddings", desc: "IDs → dense vectors" },
  { n: 4, slug: "positional-embeddings", title: "Position", desc: "Add order information" },
  { n: 5, slug: "predict-next-token", title: "Predict next token", desc: "The core task" },
  { n: 6, slug: "self-attention", title: "Self-attention", desc: "Tokens exchange context" },
  { n: 8, slug: "transformer-block", title: "Transformer block", desc: "Attention + MLP, stacked" },
  { n: 9, slug: "logits-softmax", title: "Logits → softmax", desc: "Scores → probabilities" },
  { n: 10, slug: "loss", title: "Loss", desc: "How wrong are we?" },
  { n: 11, slug: "gradient-descent", title: "Backprop + optimizer", desc: "Gradients, Adam, updates" },
  { n: 14, slug: "training-loop", title: "Training loop", desc: "Repeat until the loss is low" },
  { n: 15, slug: "generate-text", title: "Generate", desc: "Sample new text" },
];

export default function Pipeline() {
  const [idx, setIdx] = useState(-1);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    if (idx >= STAGES.length - 1) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => setIdx((i) => i + 1), 700);
    return () => clearTimeout(t);
  }, [playing, idx]);

  return (
    <Vis
      title="The whole pipeline, in one view"
      caption={
        <>
          Press <strong>Play</strong> to trace a token from raw text all the way to a new word.
          Click any stage to jump back to its lesson.
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

      <div className="stack" style={{ gap: 6, marginTop: 8 }}>
        {STAGES.map((s, i) => {
          const active = i === idx;
          return (
            <Link
              key={s.slug}
              href={`/learn/${s.slug}`}
              style={{
                display: "flex",
                gap: 12,
                alignItems: "center",
                padding: "9px 14px",
                borderRadius: 10,
                border: `1.5px solid ${active ? "var(--accent)" : "var(--border)"}`,
                background: active ? "var(--accent-soft)" : "var(--surface)",
                textDecoration: "none",
                color: "var(--text)",
                transition: "all .12s",
              }}
            >
              <span className="mono" style={{ fontWeight: 700, color: active ? "var(--accent)" : "var(--text-3)", minWidth: 24 }}>
                {formatLessonNumber(s.n)}
              </span>
              <span style={{ fontWeight: 600 }}>{s.title}</span>
              <span className="small faint" style={{ marginLeft: "auto" }}>
                {s.desc}
              </span>
            </Link>
          );
        })}
      </div>

      <div className="callout callout-key" style={{ marginTop: 14 }}>
        <p className="callout-title">🔑 The one-sentence summary</p>
        <p style={{ margin: 0 }}>
          An LLM is just a function that turns a sequence of tokens into a probability for the next
          token — and <strong>training</strong> means tweaking its weights (via gradients and an
          optimizer) so those probabilities match real text, over and over again.
        </p>
      </div>
    </Vis>
  );
}
