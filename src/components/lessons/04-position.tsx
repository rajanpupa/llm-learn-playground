import { Callout, Section } from "@/components/ui";
import PositionalViz from "@/components/viz/PositionalViz";

export default function Lesson04() {
  return (
    <div className="prose">
      <Section title="The same word in a different place is a different thing">
        <p>
          Consider <span className="mono">the cat sat</span> versus <span className="mono">the sat
          cat</span>. The exact same tokens, in a different order, mean something different (or
          nothing at all). But if each token only gets its <em>embedding</em>, then{" "}
          <span className="mono">sat</span> looks identical whether it&apos;s first or last. The
          model would have no idea about order.
        </p>
        <p>
          The fix: add a second vector that depends only on <strong>where</strong> the token is — a{" "}
          <strong>position vector</strong>. The input to the model becomes{" "}
          <strong>token embedding + position embedding</strong>.
        </p>
      </Section>

      <PositionalViz />

      <Section title="How positions get their own vectors">
        <p>
          Each of the (up to) 8 positions in our window has a learned vector. Position 0 always adds
          the same vector, position 4 always adds another. This way, <span className="mono">t</span>{" "}
          at position 0 and <span className="mono">t</span> at position 4 become different numbers —
          even though their token embedding is identical.
        </p>
        <p>
          Some models learn these position vectors; others use a fixed sine-wave formula (which has
          a nice property: position 2&apos;s vector is close to position 3&apos;s). Either way, the
          goal is the same — <strong>stamp every token with where it lives</strong>.
        </p>
      </Section>

      <Callout kind="key" title="Input = meaning + position">
        The model&apos;s very first layer receives <strong>two vectors added together</strong>: one
        that says <em>what</em> the token is, and one that says <em>where</em> it is. That
        combination is the raw material everything else is built from.
      </Callout>
    </div>
  );
}
