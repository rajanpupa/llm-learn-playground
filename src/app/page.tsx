import Link from "next/link";
import { LESSONS, formatLessonNumber } from "@/lib/lessons";

const GROUPS: { label: string; blurb: string }[] = [
  { label: "Foundations", blurb: "Turn raw text into numbers a model can use." },
  { label: "The Architecture", blurb: "The transformer — how a model reads context." },
  { label: "Learning", blurb: "How a model actually gets better, step by step." },
  { label: "Inference & Wrap-up", blurb: "Turning a trained model into new text." },
];

export default function Home() {
  return (
    <main>
      <header className="landing-hero">
        <span className="badge" style={{ marginBottom: 16 }}>
          Interactive course
        </span>
        <h1>How an LLM is trained</h1>
        <p className="tagline">
          A visual, step-by-step guide — from raw text to a model that writes.
          No math background needed. One tiny running example the whole way through.
        </p>
        <div className="row" style={{ justifyContent: "center", gap: 12 }}>
          <Link className="btn btn-primary" href={`/learn/${LESSONS[0].slug}`}>
            Start Lesson 01 →
          </Link>
          <Link className="btn btn-accent-soft" href="/playground">
            🧪 Open the Playground
          </Link>
          <a className="btn btn-ghost" href="#toc">
            Browse all lessons
          </a>
        </div>
      </header>

      <section className="landing-toc" id="toc">
        {GROUPS.map((group) => {
          const lessons = LESSONS.filter((l) => l.category === group.label);
          if (lessons.length === 0) return null;
          return (
            <div className="landing-group" key={group.label}>
              <h2>{group.label}</h2>
              <p className="muted small" style={{ marginTop: -4 }}>
                {group.blurb}
              </p>
              {lessons.map((l) => (
                <Link className="landing-item" href={`/learn/${l.slug}`} key={l.slug}>
                  <span className="num">{formatLessonNumber(l.number)}</span>
                  <span>
                    <span className="title">{l.title}</span>
                    <span className="desc">
                      {" "}
                      — {l.summary}
                    </span>
                  </span>
                </Link>
              ))}
            </div>
          );
        })}
      </section>
    </main>
  );
}
