import { Callout, Section, Eq } from "@/components/ui";
import PredictionTask from "@/components/viz/PredictionTask";
import { DATASET, BLOCK_SIZE } from "@/lib/corpus";

export default function Lesson05() {
  return (
    <div className="prose">
      <Section title="One job, repeated forever">
        <p>
          We now know how to turn text into meaningful numbers. The next question is:{" "}
          <strong>what do we ask the model to do?</strong> The answer is deceptively simple — given
          some text, <strong>predict the next token</strong>.
        </p>
        <p>
          To make this a concrete training problem, we slide a window of <Eq>{BLOCK_SIZE}</Eq>{" "}
          tokens across our corpus. The window is the <strong>context</strong> (the input); the very
          next token is the <strong>target</strong> (the answer). Our tiny corpus yields{" "}
          <strong>{DATASET.length}</strong> such training examples.
        </p>
      </Section>

      <PredictionTask />

      <Section title="Why this is the perfect task">
        <p>
          Predicting the next token is <strong>self-supervised</strong>: the &ldquo;answer&rdquo; is
          already sitting in the text — no human has to label anything. You just slide a window and
          read off the next character. That&apos;s why models can train on the entire internet.
        </p>
        <p>
          And it&apos;s shockingly powerful: to predict the next word well, a model must learn
          grammar, facts, and reasoning. &ldquo;Next-token prediction&rdquo; is the engine behind
          everything an LLM does — writing, translating, even coding.
        </p>
      </Section>

      <Callout kind="key" title="Input → output, concretely">
        Context <span className="mono">the cat ␣</span> (8 tokens) in, guess{" "}
        <span className="mono">s</span> out. The model returns a probability for <em>every</em> token
        in the vocabulary; training rewards it when the true next token gets high probability.
      </Callout>
    </div>
  );
}
