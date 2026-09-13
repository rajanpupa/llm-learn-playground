import { Callout, Section, Eq } from "@/components/ui";
import OptimizerViz from "@/components/viz/OptimizerViz";

export default function Lesson13() {
  return (
    <div className="prose">
      <Section title="Raw gradients aren't quite enough">
        <p>
          Backprop gives us the gradient — but we don&apos;t just subtract it. Real training uses a
          smarter <strong>optimizer</strong> to turn gradients into weight updates. The default for
          LLMs is <strong>Adam</strong> (technically AdamW).
        </p>
        <p>
          Adam keeps a running <strong>momentum</strong> (an average of past gradients, so updates
          keep pushing in a consistent direction) and an <strong>adaptive step size</strong> (each
          weight learns how big its typical gradient is, and normalizes by it). Together they make
          training fast and stable.
        </p>
        <p>
          On top of Adam, practitioners add two safety rails: <strong>gradient clipping</strong>{" "}
          (cap the size of any single update) and a <strong>learning-rate schedule</strong> (warm up,
          then decay).
        </p>
      </Section>

      <OptimizerViz />

      <Section title="Putting the pieces together">
        <p>
          The full update rule, in words: <Eq>weight ← weight − lr × Adam(momentum, gradient)</Eq>{" "}
          — with the gradient first clipped to a safe size. You rarely need to remember the exact
          math; what matters is that the optimizer is the engine that turns &ldquo;here&apos;s the
          error&rdquo; into &ldquo;here&apos;s a smarter set of weights.&rdquo;
        </p>
      </Section>

      <Callout kind="key" title="Optimizer = the driver">
        Gradients tell you the direction; the optimizer decides the <em>speed, smoothness, and
        safety</em> of each step. A good optimizer is a big part of why modern LLMs train at all.
      </Callout>
    </div>
  );
}
