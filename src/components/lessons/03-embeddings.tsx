import { Callout, Section } from "@/components/ui";
import EmbeddingViz from "@/components/viz/EmbeddingViz";
import { EMBED_DIM } from "@/lib/corpus";

export default function Lesson03() {
  return (
    <div className="prose">
      <Section title="From sparse one-hot to dense vectors">
        <p>
          A one-hot vector is {`“`}all zeros except one 1{`”`} — it tells you <em>which</em>{" "}
          token, but nothing about what the token is <em>like</em>. So we replace it with a{" "}
          <strong>dense embedding</strong>: a short vector of {EMBED_DIM} real numbers that the
          model <em>learns</em> for each token.
        </p>
        <p>
          Practically, an embedding is just a <strong>lookup table</strong>. Token id 0 →
          row 0 of the table, token id 5 → row 5, and so on. Each row is that token&apos;s vector.
          This table is called the <strong>embedding matrix</strong>.
        </p>
      </Section>

      <EmbeddingViz />

      <Section title="Why a vector instead of a single number?">
        <p>
          A vector gives the model {EMBED_DIM} &ldquo;knobs&rdquo; to describe each token. Over
          training, tokens that behave similarly get pushed toward similar vectors — so{" "}
          <span className="mono">cat</span> and <span className="mono">dog</span> end up
          &ldquo;near&rdquo; each other in embedding space (they both appear after{" "}
          <span className="mono">the</span>), while <span className="mono">sat</span> and{" "}
          <span className="mono">ran</span> share a &ldquo;verb-ish&rdquo; region.
        </p>
        <p>
          Right now, at initialization, the vectors are random — so the 2D plot looks like a
          scatter of unrelated points. <strong>Training is what arranges them into meaning.</strong>
        </p>
      </Section>

      <Callout kind="key" title="The key mental model">
        <strong>Embedding = learned meaning.</strong> The same token always maps to the same vector,
        and the model gradually tunes each vector so that distance in embedding space reflects
        similarity in behavior.
      </Callout>
    </div>
  );
}
