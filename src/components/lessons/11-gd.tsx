import { Callout, Section, Eq } from "@/components/ui";
import LossLandscape from "@/components/viz/LossLandscape";

export default function Lesson11() {
  return (
    <div className="prose">
      <Section title="The loss is a landscape">
        <p>
          Think of the model&apos;s millions of weights as coordinates, and the loss as the{" "}
          <strong>height of the ground</strong> at those coordinates. One set of weights = one spot
          on the landscape. Training is the search for the <strong>lowest valley</strong>.
        </p>
        <p>
          We can&apos;t see a million dimensions, but the idea is identical in 2D: at any point, the{" "}
          <strong>gradient</strong> points in the direction of <em>steepest uphill</em>. So to go
          down, we take a small step <em>opposite</em> the gradient:
        </p>
        <p style={{ textAlign: "center" }}>
          <Eq>w ← w − lr × ∇loss</Eq>
        </p>
        <p>
          The <strong>learning rate</strong> (<span className="mono">lr</span>) controls the step
          size.
        </p>
      </Section>

      <LossLandscape />

      <Section title="Step size is a balancing act">
        <ul>
          <li>
            Too <strong>small</strong> → you inch along forever (slow, expensive training).
          </li>
          <li>
            Too <strong>big</strong> → you overshoot and bounce around the valley (or fly off
            entirely).
          </li>
          <li>
            Just right → you descend steadily to the bottom.
          </li>
        </ul>
        <p>
          Try it above: drop a point, run with a small <span className="mono">lr</span>, then reset
          and crank the <span className="mono">lr</span> up to watch it overshoot.
        </p>
      </Section>

      <Callout kind="key" title="Gradient descent in one sentence">
        Repeatedly nudge every weight a tiny bit in the direction that <strong>reduces the
        loss</strong>, and the model gets better. The gradient tells us that direction; the learning
        rate tells us how far to nudge.
      </Callout>
    </div>
  );
}
