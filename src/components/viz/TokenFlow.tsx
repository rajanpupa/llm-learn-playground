"use client";

import { useState } from "react";
import { FIRST_LINE, tokenize, tokenDisplay } from "@/lib/corpus";
import { TokenDot, Vis, tokenColor } from "@/components/ui";

export default function TokenFlow() {
  const ids = tokenize(FIRST_LINE);
  const [hover, setHover] = useState(-1);

  return (
    <Vis
      title="Tokenizing our running example"
      caption={
        <>
          Every character — <strong>including the space</strong> — becomes one token. Hover a
          character to see it split out, then mapped to a number. The same letter (<code>t</code>)
          always maps to the same token id.
        </>
      }
    >
      {/* Raw text */}
      <div className="row" style={{ gap: 2, marginBottom: 14 }}>
        <span className="faint small" style={{ minWidth: 92 }}>
          Raw text
        </span>
        {FIRST_LINE.split("").map((c, i) => (
          <span
            key={i}
            className="mono"
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(-1)}
            style={{
              padding: "4px 6px",
              borderRadius: 6,
              fontSize: "1.15rem",
              background: hover === i ? "var(--accent-soft)" : "var(--surface-2)",
              border: `1px solid ${hover === i ? "var(--accent-soft-2)" : "var(--border)"}`,
              cursor: "default",
              transition: "background .1s",
            }}
          >
            {c === " " ? "␣" : c}
          </span>
        ))}
      </div>

      {/* Tokens with ids */}
      <div className="row" style={{ gap: 4, alignItems: "flex-start" }}>
        <span className="faint small" style={{ minWidth: 92, paddingTop: 8 }}>
          Tokens →
        </span>
        {ids.map((id, i) => (
          <div
            key={i}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(-1)}
            style={{
              textAlign: "center",
              padding: 6,
              borderRadius: 8,
              border: `1.5px solid ${hover === i ? tokenColor(id) : "transparent"}`,
              background: hover === i ? "var(--surface-2)" : "transparent",
              cursor: "default",
            }}
          >
            <div
              className="chip"
              style={{
                fontSize: "1.15rem",
                borderColor: "transparent",
                background: "var(--surface-3)",
              }}
            >
              {tokenDisplay(id)}
            </div>
            <div
              className="mono small"
              style={{ color: tokenColor(id), fontWeight: 700, marginTop: 4 }}
            >
              {id}
            </div>
          </div>
        ))}
      </div>

      <div className="row" style={{ marginTop: 14, gap: 6 }}>
        <TokenDot id={3} />
        <span className="small muted">
          The space is token <strong>3</strong>. A newline would be token <strong>7</strong> (
          <span className="mono">⏎</span>) — punctuation and whitespace are tokens too.
        </span>
      </div>
    </Vis>
  );
}
