import { Callout, Section } from "@/components/ui";
import VocabTable from "@/components/viz/VocabTable";
import { VOCAB_SIZE } from "@/lib/corpus";

export default function Lesson02() {
  return (
    <div className="prose">
      <Section title="Every distinct token gets an integer">
        <p>
          Our corpus uses only <strong>{VOCAB_SIZE} distinct characters</strong> in total. We list
          them all and assign each one an integer id, from 0 upward. This list is the{" "}
          <strong>vocabulary</strong> — the set of everything the model can possibly read or write.
        </p>
        <p>
          To hand a token to the model, we don&apos;t hand it the number directly as a single value.
          We hand it a <strong>one-hot vector</strong>: a list of {VOCAB_SIZE} numbers that are all{" "}
          <code>0</code>, except for a single <code>1</code> at that token&apos;s id. It&apos;s like a
          row of switches with exactly one flipped on.
        </p>
      </Section>

      <VocabTable />

      <Section title="Reading the one-hot vector">
        <p>
          For example, the token <span className="mono">t</span> has id <strong>0</strong>, so its
          one-hot is <span className="mono">[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]</span>.
          The space (<span className="mono">␣</span>) has id <strong>3</strong>, so its one-hot puts
          the <code>1</code> in position 3.
        </p>
        <p>
          One-hot vectors are simple but wasteful — {VOCAB_SIZE} numbers to say one thing. In the
          next lesson we&apos;ll compress them into something far more useful.
        </p>
      </Section>

      <Callout kind="key" title="Why numbers, not letters?">
        Every operation a neural network does is arithmetic (multiply and add). Arithmetic needs
        numbers. The vocabulary is the fixed dictionary that translates between the human world of
        characters and the model&apos;s world of integers.
      </Callout>

      <Callout kind="tip" title="Real vocabularies are bigger">
        Our vocabulary has {VOCAB_SIZE} tokens. A real LLM&apos;s vocabulary has tens of thousands
        (e.g. ~50,000 for many GPT models). The idea is identical — just more rows.
      </Callout>
    </div>
  );
}
