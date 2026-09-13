import { Callout, Section } from "@/components/ui";
import TokenFlow from "@/components/viz/TokenFlow";
import { CORPUS_LINES } from "@/lib/corpus";

export default function Lesson01() {
  return (
    <div className="prose">
      <Section title="A model can't read letters — it reads numbers">
        <p>
          A language model is, at its core, a machine that reads text and predicts what comes next.
          But a computer can&apos;t actually &ldquo;read&rdquo; letters — it only does arithmetic on
          numbers. So the very first step in training an LLM is to <strong>turn text into
          numbers</strong>.
        </p>
        <p>
          We&apos;ll learn the whole pipeline on one tiny example that we carry through every
          lesson. It&apos;s a four-line &ldquo;corpus&rdquo; (a fancy word for &ldquo;our training
          text&rdquo;):
        </p>
        <pre
          style={{
            background: "var(--surface-2)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            padding: "12px 16px",
            fontFamily: "var(--font-mono)",
            lineHeight: 1.8,
          }}
        >
          {CORPUS_LINES.join("\n")}
        </pre>
        <p>
          The smallest unit we&apos;ll feed the model is a <strong>single character</strong>. That
          means every letter, every space, and every newline is its own token.
        </p>
      </Section>

      <TokenFlow />

      <Callout kind="key" title="What is a token?">
        A <strong>token</strong> is the smallest unit a model reads. Big models like GPT use{" "}
        <em>subword</em> tokens (so &ldquo;playing&rdquo; might split into &ldquo;play&rdquo; +
        &ldquo;ing&rdquo;). We use single characters instead, so you can see <em>every</em> number
        in action. Everything else works exactly the same way.
      </Callout>

      <Callout kind="tip" title="Why this matters">
        Tokenization is the foundation: if the model sees the wrong numbers, nothing downstream can
        be right. Once you understand characters → numbers, the rest is just layers of arithmetic.
      </Callout>
    </div>
  );
}
