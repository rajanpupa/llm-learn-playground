import { Callout, Section, Eq } from "@/components/ui";
import SoftmaxViz from "@/components/viz/SoftmaxViz";
import { VOCAB_SIZE } from "@/lib/corpus";

export default function Lesson09() {
  return (
    <div className="prose">
      <Section title="The model speaks in scores">
        <p>
          After the transformer blocks, we have one context-rich vector per token. To predict the
          next token, we multiply the <em>last</em> token&apos;s vector by a matrix and get{" "}
          <strong>{VOCAB_SIZE} raw scores</strong> — one per vocabulary token. These scores are
          called the <strong>logits</strong>.
        </p>
        <p>
          Logits are unconstrained: they can be negative, huge, or tiny, and they don&apos;t sum to
          anything. To read them as &ldquo;how likely is each next token,&rdquo; we apply the{" "}
          <strong>softmax</strong> function:
        </p>
        <p style={{ textAlign: "center" }}>
          <Eq>softmax(z)_i = e^(z_i) / Σ e^(z_j)</Eq>
        </p>
        <p>
          Softmax does two things at once: it makes every number <strong>positive</strong> (via{" "}
          <span className="mono">e^z</span>) and makes them <strong>sum to 1</strong> (via the
          divide). The result is a proper probability distribution.
        </p>
      </Section>

      <SoftmaxViz />

      <Section title="Scores → probabilities, note the ordering">
        <p>
          Crucially, softmax <em>preserves order</em>: the highest logit always becomes the highest
          probability. So when we later sample, we mostly pick the token with the biggest score —
          but the full distribution lets us also sample less-likely (more creative) tokens.
        </p>
      </Section>

      <Callout kind="key" title="Logits vs probabilities">
        <strong>Logits</strong> = raw model output (any numbers). <strong>Probabilities</strong> =
        logits after softmax (positive, sum to 1). The model is judged — and trained — on the
        probabilities.
      </Callout>
    </div>
  );
}
