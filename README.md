# LLM Learn — How an LLM is trained

An interactive, visual, step-by-step course on how a large language model is trained —
from raw text to a model that writes. Built as a Next.js app with hand-rolled React + SVG
visualizations (no chart libraries).

One tiny running example — a 4-line character corpus and a genuinely-trainable 8-dimension
transformer called **ToyGPT** — is threaded through every lesson so concepts build on each other.

## Run it

```bash
npm install
npm run dev        # http://localhost:3000
```

Or build a fully-static site (no server needed):

```bash
npm run build      # emits to out/
# open out/index.html directly, or serve it:
npx serve out
```

## The 16 lessons

1. **Text into Tokens** — split text into characters
2. **The Vocabulary & Token IDs** — map tokens to one-hot vectors
3. **Embeddings** — token ids become dense vectors
4. **Position** — stamp each token with where it lives
5. **Predict the Next Token** — the core task
6. **Self-Attention** — Query/Key/Value and attention weights
7. **Multi-Head Attention** — several lenses in parallel
8. **The Transformer Block** — attention + MLP + residuals
9. **Logits & Softmax** — scores become probabilities
10. **Loss** — cross-entropy measures mistakes
11. **Gradient Descent** — walking downhill on the loss landscape
12. **Backpropagation** — the chain rule for every weight
13. **The Optimizer** — Adam, gradient clipping, LR schedules
14. **The Training Loop** — watch ToyGPT train live in your browser
15. **Generating Text** — temperature & top-k sampling
16. **Recap** — the whole pipeline in one view

## The Playground

A separate [`/playground`](src/app/playground/page.tsx) page lets you train a tiny model on **your
own** text — paste anything (code, lyrics, a recipe, another language) and the model adapts to it.
Four steps:

1. **Text** — paste text or upload a `.txt`; it's split into tokens and turned into sliding-window
   training examples.
2. **Weights** — inspect the token-embedding matrix, hand-edit any cell, randomize, reset, or
   download/upload a trained model as JSON.
3. **Train** — step through training (1/10/200 steps, or any number) and watch the loss curve drop.
4. **Predict** — predict the next token, or generate 40 tokens with a temperature slider.

Three **tokenizers** are available (chosen in step 1; everything else adapts):

- **Character** — each character is a token.
- **Byte-pair encoding (BPE)** — subword pieces learned from your text (16 merge steps).
- **Word** — each whitespace-delimited word is a token.

Every step has an expandable **"What's happening here?"** section explaining what the step does,
whether it's randomized or deterministic, how the calculation and backprop work, and links to the
relevant lesson.

## How it's built

- `src/lib/corpus.ts` — the running example (corpus, vocabulary, dataset)
- `src/lib/model.ts` — ToyGPT: forward pass, sampling (real transformer)
- `src/lib/train.ts` — backprop, AdamW optimizer, training loop
- `src/components/viz/` — the interactive visualizations
- `src/components/lessons/` — one content component per lesson

The training loop in lesson 14 is a **real** transformer with real backprop and the Adam
optimizer — it trains in the browser and its loss curve drops live.
