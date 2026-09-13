import { Callout, Section } from "@/components/ui";
import BackpropViz from "@/components/viz/BackpropViz";

export default function Lesson12() {
  return (
    <div className="prose">
      <Section title="The gradient for every weight, in one pass">
        <p>
          We know <em>that</em> we want to move weights to reduce the loss. But the model has
          thousands of weights — how do we know how much <em>each one</em> contributed to the
          error? That&apos;s the job of <strong>backpropagation</strong>.
        </p>
        <p>
          The trick is the <strong>chain rule</strong> from calculus: if the loss depends on the
          output, the output on the logits, the logits on the attention, and so on, then the
          influence of an early weight is just the <em>product</em> of all the little influences
          along the way.
        </p>
        <p>
          So we run the network <strong>backward</strong> — from the loss toward the input — and at
          each step multiply the incoming gradient by that layer&apos;s local derivative. By the
          time we reach the embeddings, we have a gradient for <strong>every single weight</strong>,
          all in a single backward pass.
        </p>
      </Section>

      <BackpropViz />

      <Section title="Forward measures, backward fixes">
        <p>
          Notice the beautiful symmetry: the <strong>forward pass</strong> computes a prediction and
          a loss; the <strong>backward pass</strong> retraces the exact same path in reverse,
          distributing blame. Then the optimizer (next lesson) applies the fixes.
        </p>
        <p>
          This is why the transformer was a revolution: because every operation in it is
          differentiable, backprop can flow through the <em>entire</em> stack at once — no
          hand-crafting of &ldquo;which rule to adjust.&rdquo;
        </p>
      </Section>

      <Callout kind="key" title="Why it's efficient">
        Backprop computes the gradient for <em>all</em> weights with roughly the same cost as the
        forward pass itself — one forward, one backward. It&apos;s the workhorse that makes training
        huge networks feasible.
      </Callout>
    </div>
  );
}
