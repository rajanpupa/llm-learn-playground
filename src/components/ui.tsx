import type { ReactNode } from "react";
import { tokenDisplay } from "@/lib/corpus";

/** CSS variable for a token's stable categorical color. */
export function tokenColor(id: number): string {
  return `var(--tok-${id % 13})`;
}

/** A small colored dot for a token. */
export function TokenDot({ id }: { id: number }) {
  return <span className="token-dot" style={{ background: tokenColor(id) }} />;
}

/** A chip showing a single token (with its color dot). */
export function TokenChip({
  id,
  active = false,
  label,
}: {
  id: number;
  active?: boolean;
  label?: string;
}) {
  return (
    <span className={`chip ${active ? "accent" : ""}`} style={{ gap: 6 }}>
      <TokenDot id={id} />
      <span>{label ?? tokenDisplay(id)}</span>
    </span>
  );
}

/** A sequence of tokens as chips. */
export function TokenRow({ ids, activeIndex = -1 }: { ids: number[]; activeIndex?: number }) {
  return (
    <span className="row" style={{ gap: 5 }}>
      {ids.map((id, i) => (
        <TokenChip key={i} id={id} active={i === activeIndex} />
      ))}
    </span>
  );
}

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="prose">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export function Callout({
  kind = "key",
  title,
  children,
}: {
  kind?: "key" | "tip" | "warn";
  title: string;
  children: ReactNode;
}) {
  const cls = kind === "key" ? "callout-key" : kind === "tip" ? "callout-tip" : "callout-warn";
  const glyph = kind === "key" ? "🔑" : kind === "tip" ? "💡" : "⚠️";
  return (
    <div className={`callout ${cls}`}>
      <p className="callout-title">
        <span>{glyph}</span> {title}
      </p>
      <div>{children}</div>
    </div>
  );
}

/** Wrapper for an interactive visualization. */
export function Vis({
  title,
  caption,
  children,
}: {
  title?: string;
  caption?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="vis">
      {title && <div className="card-title">{title}</div>}
      {children}
      {caption && <div className="vis-caption">{caption}</div>}
    </div>
  );
}

/** Inline code formula. */
export function Eq({ children }: { children: ReactNode }) {
  return <span className="eq">{children}</span>;
}
