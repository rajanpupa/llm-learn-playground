import { Callout, Section } from "@/components/ui";
import Pipeline from "@/components/viz/Pipeline";

export default function Lesson16() {
  return (
    <div className="prose">
      <Section title="You now understand how an LLM is trained">
        <p>
          Let&apos;s step back and see the whole journey at once. In just 15 lessons you&apos;ve
          traced a single character through every stage of a real — if tiny — language model.
        </p>
        <p>
          The real thing (GPT-4, Claude, and friends) is the <em>same pipeline</em>, scaled up:
          a larger vocabulary, thousands of dimensions instead of 8, dozens of layers instead of 1,
          trillions of training tokens instead of 47 characters, and months of GPU time instead of
          one browser click.
        </p>
      </Section>

      <Pipeline />

      <Section title="Where to go from here">
        <ul>
          <li>
            <strong>Scale is the magic.</strong> The architecture you learned was mostly settled by
            2018; the last few years have been about data, compute, and alignment.
          </li>
          <li>
            <strong>Play with the real thing.</strong> The reference repo{" "}
            <span className="mono">llm-from-scratch</span> trains a slightly bigger character GPT on
            Shakespeare — the same code, just bigger numbers.
          </li>
          <li>
            <strong>Revisit any lesson.</strong> The sidebar has them all — go back to attention or
            backprop whenever a detail feels fuzzy.
          </li>
        </ul>
      </Section>

      <Callout kind="tip" title="The most important sentence in this course">
        An LLM is a function from a sequence of tokens to a probability for the next token — and
        <strong> training</strong> is gradient descent on a loss that compares those probabilities to
        real text. Everything else is detail.
      </Callout>
    </div>
  );
}
