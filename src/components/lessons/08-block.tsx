import { Callout, Section } from "@/components/ui";
import TransformerDiagram from "@/components/viz/TransformerDiagram";

export default function Lesson08() {
  return (
    <div className="prose">
      <Section title="Attention + a think step, wrapped in a block">
        <p>
          Self-attention is only half the story. After tokens exchange context, each token also
          needs a moment to <em>think on its own</em> — that&apos;s a small feed-forward network
          (the <strong>MLP</strong>), applied identically to every token. Attention mixes{" "}
          <em>between</em> tokens; the MLP computes <em>within</em> each token.
        </p>
        <p>
          These two pieces, plus <strong>layer normalization</strong> (which keeps numbers at a
          stable scale) and <strong>residual connections</strong> (which let the original signal
          flow straight through), form one <strong>transformer block</strong>.
        </p>
      </Section>

      <TransformerDiagram />

      <Section title="Stack them high">
        <p>
          One block gives each token one round of &ldquo;look around, then think.&rdquo; Real models
          stack many blocks: GPT-2 had 12, GPT-3 had 96. After each block, the token vectors get a
          little more context-aware — first local patterns, then grammar, then meaning, then
          reasoning.
        </p>
        <p>
          Our ToyGPT uses a single block (and {`8`} dimensions), but it&apos;s the <em>same</em>{" "}
          architecture — just tiny. Everything you&apos;ve seen so far is a real, working GPT.
        </p>
      </Section>

      <Callout kind="key" title="The transformer recipe">
        <strong>Embeddings + position</strong> in → repeat <strong>[attention + MLP, with
        residuals &amp; layer-norm]</strong> → a vector per token that now &ldquo;understands&rdquo;
        its context. Next, we turn those vectors into actual predictions.
      </Callout>
    </div>
  );
}
