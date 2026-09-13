import { Callout, Section } from "@/components/ui";
import { MultiHeadViz } from "@/components/viz/Attention";
import { N_HEAD } from "@/lib/corpus";

export default function Lesson07() {
  return (
    <div className="prose">
      <Section title="One head isn't enough">
        <p>
          A single attention pattern forces each token to blend <em>everything</em> it cares about
          into one summary. But real language has many simultaneous relationships — the previous
          word, the subject of the sentence, a closing quote. One number can&apos;t weigh all of
          those at once.
        </p>
        <p>
          The solution: run <strong>several attention heads in parallel</strong>, each with its own
          Query/Key/Value weights, then <strong>concatenate</strong> their outputs. Each head is
          free to learn a different &ldquo;lens&rdquo;. Our ToyGPT uses {N_HEAD} heads.
        </p>
      </Section>

      <MultiHeadViz />

      <Section title="How it fits together">
        <p>
          Each head produces its own context vector for every token. We simply glue all heads&apos;
          vectors together into one long vector, then project back down with a learned matrix.
          From the outside, multi-head attention still looks like &ldquo;tokens in → richer tokens
          out&rdquo; — the heads are just the mechanism inside.
        </p>
        <p>
          In a big model like GPT-3, there are <strong>96 heads</strong> across 96 layers — a lot of
          lenses. Together they capture an enormous amount of linguistic structure.
        </p>
      </Section>

      <Callout kind="key" title="Why &quot;multi-head&quot; matters">
        More heads = more distinct relationships the model can track <em>at the same time</em>,
        without them interfering with each other. It&apos;s the difference between one person trying
        to do five jobs, and five specialists.
      </Callout>
    </div>
  );
}
