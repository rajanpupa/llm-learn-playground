import Link from "next/link";
import Playground from "@/components/Playground";

export default function PlaygroundPage() {
  return (
    <main style={{ maxWidth: 960, margin: "0 auto", padding: "40px 24px 96px" }}>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
        <p className="lesson-eyebrow" style={{ margin: 0 }}>
          Playground
        </p>
        <Link href="/" className="btn btn-ghost btn-sm">
          ← Back to course
        </Link>
      </div>
      <h1 className="lesson-title">Train your own tiny model</h1>
      <p className="lesson-summary">
        Paste your own text, pick a vector size, peek at (and hand-edit) the weights, then train it
        step by step and predict what comes next.
      </p>
      <Playground />
    </main>
  );
}
