import { Callout, Section, Eq } from "@/components/ui";
import AttentionViz from "@/components/viz/Attention";

export default function Lesson06() {
  return (
    <div className="prose">
      <Section title="Tokens need to talk to each other">
        <p>
          So far each token has been processed in isolation. But meaning comes from context: in{" "}
          <span className="mono">the cat sat</span>, the word <span className="mono">cat</span>{" "}
          should look back at <span className="mono">the</span> to understand it&apos;s a noun.
          That&apos;s what <strong>self-attention</strong> does — it lets every token gather
          information from the tokens before it.
        </p>
        <p>
          Each token produces three vectors: a <strong>Query</strong> (&ldquo;what am I looking
          for?&rdquo;), a <strong>Key</strong> (&ldquo;what do I contain?&rdquo;), and a{" "}
          <strong>Value</strong> (&ldquo;what do I contribute if attended?&rdquo;). A token&apos;s
          attention to another token is the similarity of its Query to that token&apos;s Key:
        </p>
        <p style={{ textAlign: "center" }}>
          <Eq>score = (Query · Key) / √d</Eq> → <Eq>softmax</Eq> → <Eq>output = Σ weight × Value</Eq>
        </p>
        <p>
          The result for each token is a <strong>weighted average of the Values</strong> of the
          tokens it attended to.
        </p>
      </Section>

      <AttentionViz />

      <Section title="The causal mask">
        <p>
          Notice the empty top-right triangle in the heatmap. When predicting the <em>next</em>{" "}
          token, the model must never peek at the future — that would be cheating. So attention is{" "}
          <strong>causal</strong>: a token can only look at itself and earlier tokens. The future is
          masked out (set to zero probability).
        </p>
      </Section>

      <Callout kind="key" title="Attention in one line">
        Attention is a <strong>learned way to move information between tokens</strong>: each token
        decides, via Query·Key similarity, how much of every earlier token&apos;s Value to mix into
        its own representation.
      </Callout>
    </div>
  );
}
