import { Callout, Section, Eq } from "@/components/ui";
import LossViz from "@/components/viz/LossViz";

export default function Lesson10() {
  return (
    <div className="prose">
      <Section title="A single number for &quot;how wrong?&quot;">
        <p>
          The model outputs a probability distribution over the next token. The training data tells
          us the <em>true</em> next token. We need one number that says how far apart they are —
          that number is the <strong>loss</strong>, and the standard choice is{" "}
          <strong>cross-entropy</strong>.
        </p>
        <p>
          For a single prediction, cross-entropy is simply the negative log of the probability the
          model assigned to the <em>correct</em> token:
        </p>
        <p style={{ textAlign: "center" }}>
          <Eq>loss = -log p(correct)</Eq>
        </p>
        <ul>
          <li>
            Model is confident and right (<span className="mono">p≈1</span>) → loss ≈ <strong>0</strong>.
          </li>
          <li>
            Model is confident and wrong (<span className="mono">p≈0</span>) → loss is{" "}
            <strong>huge</strong>.
          </li>
          <li>
            Model is unsure (<span className="mono">p≈0.5</span>) → loss is moderate.
          </li>
        </ul>
        <p>
          We average this over every position in every training example to get the model&apos;s
          overall loss.
        </p>
      </Section>

      <LossViz />

      <Section title="The loss is the training signal">
        <p>
          Everything that follows — gradients, updates, the whole training loop — exists for one
          purpose: <strong>make the loss smaller</strong>. A lower loss means the model assigns
          higher probability to the true next tokens, which means it has learned the patterns in
          the data.
        </p>
      </Section>

      <Callout kind="key" title="Memorize this">
        The <strong>loss</strong> is a differentiable score of how bad the model is. Training =
        repeatedly adjusting weights so the loss goes <strong>down</strong>. When people show a
        &ldquo;loss curve&rdquo; going down and to the right — that curve <em>is</em> learning.
      </Callout>
    </div>
  );
}
