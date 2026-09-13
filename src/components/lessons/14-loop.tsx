import { Callout, Section } from "@/components/ui";
import TrainingLoop from "@/components/viz/TrainingLoop";
import { DATASET } from "@/lib/corpus";

export default function Lesson14() {
  return (
    <div className="prose">
      <Section title="Feed, measure, fix, repeat">
        <p>
          We now have every ingredient. The <strong>training loop</strong> just puts them on repeat:
        </p>
        <ol>
          <li>
            <strong>Forward</strong> — run a batch of examples through the model, get predictions.
          </li>
          <li>
            <strong>Loss</strong> — measure how wrong those predictions are.
          </li>
          <li>
            <strong>Backward</strong> — compute the gradient of every weight.
          </li>
          <li>
            <strong>Update</strong> — let the optimizer adjust the weights.
          </li>
          <li>
            <strong>Repeat</strong> — go back to step 1 with the next batch.
          </li>
        </ol>
        <p>
          One pass through the whole dataset is an <strong>epoch</strong>. Models train for many
          epochs, over <em>enormous</em> datasets, on <em>enormous</em> hardware.
        </p>
      </Section>

      <TrainingLoop />

      <Section title="A few details worth knowing">
        <ul>
          <li>
            <strong>Batches</strong> — we update using a small group of examples at once (not one,
            not all), which is both fast and stable.
          </li>
          <li>
            <strong>Loss curve</strong> — we watch it fall. It starts near <span className="mono">ln(13)
            ≈ 2.57</span> (random guessing over {`13`} tokens) and drops as the model learns.
          </li>
          <li>
            <strong>Overfitting</strong> — with only {DATASET.length} examples, ToyGPT can
            memorize the corpus; the loss dives to near zero. Real models have so much data this
            rarely happens.
          </li>
        </ul>
      </Section>

      <Callout kind="key" title="The whole point, finally">
        Everything you learned — tokens, embeddings, attention, softmax, loss, gradients, optimizer —
        is <strong>one line in a loop</strong>. Run that loop billions of times and a pile of random
        numbers turns into a model that writes coherent text.
      </Callout>
    </div>
  );
}
