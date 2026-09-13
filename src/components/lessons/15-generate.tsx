import { Callout, Section } from "@/components/ui";
import Generator from "@/components/viz/Generator";

export default function Lesson15() {
  return (
    <div className="prose">
      <Section title="Now the model writes">
        <p>
          Training is done. To generate text, we give the model a starting prompt, ask it for the
          probability distribution of the next token, <strong>sample one token</strong> from that
          distribution, append it, and repeat. This is called <strong>autoregressive</strong>{" "}
          generation — each new token becomes part of the context for the next.
        </p>
        <p>
          Why <em>sample</em> instead of always picking the most likely token? Because always
          picking the top token gives repetitive, boring text. Randomness is what makes output
          creative — and two knobs control it:
        </p>
        <ul>
          <li>
            <strong>Temperature</strong> — divide logits by <span className="mono">T</span>. Low{" "}
            <span className="mono">T</span> sharpens the distribution (more predictable, focused);
            high <span className="mono">T</span> flattens it (more random, diverse).
          </li>
          <li>
            <strong>Top-k</strong> — keep only the <span className="mono">k</span> most likely
            tokens and sample among those, cutting off the long tail of nonsense.
          </li>
        </ul>
      </Section>

      <Generator />

      <Section title="The temperature dial in action">
        <p>
          At <span className="mono">temp = 0.2</span> the model almost always picks the most likely
          token — great for facts, dull for stories. At <span className="mono">temp = 1.5</span> it
          wanders — creative, but prone to gibberish. Play with the sliders and feel the difference.
        </p>
      </Section>

      <Callout kind="key" title="Generation = prediction, looped">
        There&apos;s no separate &ldquo;writing&rdquo; mechanism. An LLM writes by doing the one
        thing it was trained to do — <strong>predict the next token</strong> — over and over,
        feeding its own output back in.
      </Callout>
    </div>
  );
}
