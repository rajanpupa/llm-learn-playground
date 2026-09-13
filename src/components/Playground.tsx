"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  bpeMerges,
  tokenizeWithRules,
  buildVocabFromTokens,
  buildDatasetFromIds,
  BPE_MERGES,
  CORPUS,
  type ModelConfig,
  type Token,
  type TokenizerKind,
} from "@/lib/corpus";
import { initAdam, trainStep, evalLoss, DEFAULT_TRAIN } from "@/lib/train";
import {
  forward,
  generate,
  initParams,
  mulberry32,
  randMat,
  softmaxRow,
  type Params,
  type LossFn,
} from "@/lib/model";
import Link from "next/link";
import { tokenColor, TokenDot, Callout } from "@/components/ui";

const BLOCK = 8;
const HEADS = 2;
const DIMS = [4, 6, 8, 12];

const LOSS_OPTIONS: { value: LossFn; label: string }[] = [
  { value: "ce", label: "Cross-entropy (standard)" },
  { value: "mse", label: "Mean squared error" },
];

function charMap(vocab: Token[]): Map<string, number> {
  const m = new Map<string, number>();
  for (const t of vocab) m.set(t.char, t.id);
  return m;
}

function Disclosure({ title, children }: { title: string; children: ReactNode }) {
  return (
    <details className="disclosure">
      <summary>
        <span className="chevron">▸</span>
        <span>{title}</span>
      </summary>
      <div className="disclosure-body">{children}</div>
    </details>
  );
}

export default function Playground() {
  const [text, setText] = useState(CORPUS);
  const [embedDim, setEmbedDim] = useState(8);
  const [tokenizer, setTokenizer] = useState<TokenizerKind>("char");
  const [lossFn, setLossFn] = useState<LossFn>("ce");
  const [prompt, setPrompt] = useState("the");
  const [temp, setTemp] = useState(0.7);
  const [stepsInput, setStepsInput] = useState(100);

  const rules = useMemo(
    () => (tokenizer === "bpe" ? bpeMerges(text, BPE_MERGES) : []),
    [text, tokenizer],
  );
  const tokens = useMemo(
    () => tokenizeWithRules(text, tokenizer, rules),
    [text, tokenizer, rules],
  );
  const vocab = useMemo(() => buildVocabFromTokens(tokens), [tokens]);
  const cfg = useMemo<ModelConfig>(
    () => ({ vocabSize: vocab.length, embedDim, blockSize: BLOCK, nHead: HEADS }),
    [vocab, embedDim],
  );
  const map = useMemo(() => charMap(vocab), [vocab]);
  const tokenIds = useMemo(
    () => tokens.map((t) => map.get(t)).filter((x): x is number => x !== undefined),
    [tokens, map],
  );
  const dataset = useMemo(() => buildDatasetFromIds(tokenIds, BLOCK), [tokenIds]);
  const promptTokens = useMemo(
    () => tokenizeWithRules(prompt, tokenizer, rules),
    [prompt, tokenizer, rules],
  );
  const promptIds = useMemo(
    () => promptTokens.map((t) => map.get(t)).filter((x): x is number => x !== undefined),
    [promptTokens, map],
  );

  const paramsRef = useRef<Params | null>(null);
  const adamRef = useRef<ReturnType<typeof initAdam> | null>(null);
  const stepRef = useRef(0);
  const restoreRef = useRef<Params | null>(null);
  const cancelRef = useRef(false);
  const lossFnRef = useRef<LossFn>(lossFn);
  lossFnRef.current = lossFn;

  const textFileRef = useRef<HTMLInputElement>(null);
  const modelFileRef = useRef<HTMLInputElement>(null);

  const [loss, setLoss] = useState<number | null>(null);
  const [step, setStep] = useState(0);
  const [history, setHistory] = useState<number[]>([]);
  const [version, setVersion] = useState(0);
  const [predictions, setPredictions] = useState<{ id: number; p: number }[]>([]);
  const [generatedIds, setGeneratedIds] = useState<number[]>([]);
  const [training, setTraining] = useState(false);
  const [trainProg, setTrainProg] = useState<{ done: number; total: number } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [restoreTick, setRestoreTick] = useState(0);

  // (Re)initialize (or restore) the model whenever text / vector size / tokenizer changes.
  useEffect(() => {
    const restored = restoreRef.current;
    restoreRef.current = null;
    paramsRef.current = restored ?? initParams(cfg, 42);
    adamRef.current = initAdam(cfg);
    stepRef.current = 0;
    setLoss(dataset.length > 0 ? evalLoss(paramsRef.current, dataset, lossFnRef.current) : null);
    setStep(0);
    setHistory([]);
    setPredictions([]);
    setGeneratedIds([]);
    setVersion((v) => v + 1);
  }, [cfg, text, tokenizer, dataset, restoreTick]);

  const p = paramsRef.current;
  const ready = p != null && p.cfg === cfg;
  const dsCount = dataset.length;

  const bump = () => setVersion((v) => v + 1);

  function recomputeLoss(fn: LossFn = lossFn) {
    if (!ready || !p) return;
    setLoss(dataset.length > 0 ? evalLoss(p, dataset, fn) : null);
  }

  async function runTraining(total: number) {
    if (!ready || !p || !adamRef.current || training) return;
    const ds = dataset;
    if (ds.length === 0) return;
    cancelRef.current = false;
    setTraining(true);
    setTrainProg({ done: 0, total });
    const CHUNK = 40;
    let done = 0;
    while (done < total && !cancelRef.current) {
      const n = Math.min(CHUNK, total - done);
      for (let i = 0; i < n; i++) {
        const s = (stepRef.current * 4) % ds.length;
        const batch = Array.from({ length: Math.min(4, ds.length) }, (_, k) => ds[(s + k) % ds.length]);
        trainStep(p, adamRef.current, batch, DEFAULT_TRAIN, lossFnRef.current);
        stepRef.current += 1;
      }
      done += n;
      const l = evalLoss(p, ds, lossFnRef.current);
      setStep(stepRef.current);
      setLoss(l);
      setHistory((h) => (h.length > 240 ? [...h.slice(-160), l] : [...h, l]));
      setTrainProg({ done, total });
      bump();
      // yield so the spinner / progress bar can paint
      await new Promise((r) => setTimeout(r, 0));
    }
    setTraining(false);
    setTrainProg(null);
  }

  function randomizeEmbeddings() {
    if (!ready || !p) return;
    const rng = mulberry32(Math.floor(Math.random() * 1e9));
    p.wte = randMat(vocab.length, embedDim, rng, 0.08);
    adamRef.current = initAdam(cfg);
    stepRef.current = 0;
    setStep(0);
    recomputeLoss();
    setHistory([]);
    setPredictions([]);
    setGeneratedIds([]);
    bump();
  }

  function resetAll() {
    if (!ready) return;
    paramsRef.current = initParams(cfg, Math.floor(Math.random() * 1e9));
    adamRef.current = initAdam(cfg);
    stepRef.current = 0;
    setStep(0);
    recomputeLoss();
    setHistory([]);
    setPredictions([]);
    setGeneratedIds([]);
    bump();
  }

  function commitCell(i: number, j: number, raw: string) {
    if (!ready || !p) return;
    const v = parseFloat(raw);
    if (Number.isFinite(v)) {
      p.wte[i][j] = v;
      bump();
    }
  }

  function changeLossFn(fn: LossFn) {
    setLossFn(fn);
    lossFnRef.current = fn;
    setHistory([]);
    recomputeLoss(fn);
  }

  function predict() {
    if (!ready || !p) return;
    if (promptIds.length === 0) {
      setPredictions([]);
      return;
    }
    const ctx = promptIds.slice(-BLOCK);
    const cache = forward(p, ctx);
    const probs = softmaxRow(cache.logits[ctx.length - 1]);
    setPredictions(
      probs.map((pr, i) => ({ id: i, p: pr })).sort((a, b) => b.p - a.p).slice(0, 6),
    );
  }

  function generateText() {
    if (!ready || !p) return;
    if (promptIds.length === 0) return;
    const seedIds = promptIds.slice(-BLOCK);
    const out = generate(p, seedIds, 40, temp, 0, mulberry32(Date.now() % 100000));
    setGeneratedIds(out.slice(seedIds.length));
  }

  function downloadModel() {
    if (!ready || !p) return;
    const payload = { kind: "toygpt", text, tokenizer, lossFn, step, params: p };
    const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "toygpt-model.json";
    a.click();
    URL.revokeObjectURL(url);
    setNotice("Model downloaded as toygpt-model.json");
  }

  function onTextFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setText(String(reader.result ?? ""));
      setNotice("Text loaded");
    };
    reader.readAsText(file);
  }

  function onModelFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(String(reader.result));
        if (data?.kind !== "toygpt" || !data.params?.cfg) {
          setNotice("Invalid model file.");
          return;
        }
        restoreRef.current = data.params as Params;
        if (data.text != null) setText(data.text);
        if (data.params.cfg.embedDim) setEmbedDim(data.params.cfg.embedDim);
        if (data.lossFn) setLossFn(data.lossFn);
        if (data.tokenizer) setTokenizer(data.tokenizer);
        setRestoreTick((t) => t + 1);
        setNotice("Model loaded");
      } catch {
        setNotice("Could not parse model file.");
      }
    };
    reader.readAsText(file);
  }

  const spark = (() => {
    if (history.length < 2) return "";
    const H = 60;
    const W = 300;
    const max = Math.max(...history, 1e-6);
    return history
      .map((l, i) => `${(i / (history.length - 1)) * W},${H - (l / max) * (H - 6)}`)
      .join(" ");
  })();

  const unknownInPrompt = promptTokens.length - promptIds.length;

  const tokenizerDesc =
    tokenizer === "char"
      ? "each character (spaces and newlines included)"
      : tokenizer === "word"
        ? "each whitespace-delimited word"
        : "subword pieces via byte-pair encoding";

  return (
    <div className="stack" style={{ gap: 20 }}>
      {/* 1 · Text */}
      <div className="card card-pad">
        <div className="card-title">1 · Your text</div>
        <div className="vis-controls" style={{ marginTop: 0 }}>
          <label>
            Tokenizer:{" "}
            <select
              value={tokenizer}
              onChange={(e) => setTokenizer(e.target.value as TokenizerKind)}
              disabled={training}
              style={{
                fontFamily: "var(--font-mono)",
                padding: "6px 8px",
                borderRadius: 8,
                border: "1px solid var(--border-strong)",
                background: "var(--surface)",
                color: "var(--text)",
              }}
            >
              <option value="char">Character</option>
              <option value="bpe">Byte-pair encoding (BPE)</option>
              <option value="word">Word</option>
            </select>
          </label>
          <span className="small faint">How your text is split into tokens.</span>
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={training}
          rows={4}
          style={{
            width: "100%",
            fontFamily: "var(--font-mono)",
            fontSize: "0.95rem",
            padding: "12px",
            borderRadius: 10,
            border: "1px solid var(--border-strong)",
            background: "var(--surface-2)",
            color: "var(--text)",
            resize: "vertical",
          }}
        />
        <div className="vis-controls">
          <span className="small muted">
            {vocab.length} unique tokens · {tokens.length} tokens · {dsCount} training windows
          </span>
          <button className="btn btn-ghost btn-sm" onClick={() => textFileRef.current?.click()}>
            Upload text (.txt)
          </button>
          <input ref={textFileRef} type="file" accept=".txt,text/plain" hidden onChange={onTextFile} />
        </div>
        <div className="row" style={{ gap: 4, flexWrap: "wrap" }}>
          {vocab.map((t) => (
            <span key={t.id} className="chip" style={{ gap: 5, height: 28, fontSize: "0.85rem" }}>
              <TokenDot id={t.id} />
              {t.display}
            </span>
          ))}
        </div>

        <div className="vis-controls" style={{ marginTop: 16 }}>
          <label>
            Vector size (embedding dim):{" "}
            <select
              value={embedDim}
              onChange={(e) => setEmbedDim(Number(e.target.value))}
              disabled={training}
              style={{
                fontFamily: "var(--font-mono)",
                padding: "6px 8px",
                borderRadius: 8,
                border: "1px solid var(--border-strong)",
                background: "var(--surface)",
                color: "var(--text)",
              }}
            >
              {DIMS.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </label>
          <span className="small faint">Each token&apos;s vector has {embedDim} numbers.</span>
        </div>

        <Callout kind="tip" title="Paste any text">
          <p style={{ margin: 0 }} className="small">
            Anything goes — code, lyrics, a recipe, another language. The vocabulary, the training
            windows, and everything the model learns are rebuilt from whatever you type here, so the
            model adapts to <em>your</em> text.
          </p>
        </Callout>

        <Disclosure title="What's happening here?">
          <p>
            <strong>Vocabulary &amp; token ids.</strong> Your text is split into tokens —{" "}
            {tokenizerDesc}. Each unique token gets an integer id, numbered in order of first
            appearance.
          </p>
          <p>
            <strong>Sliding windows.</strong> The text is cut into overlapping windows of {BLOCK}{" "}
            tokens: each window&apos;s <em>input</em> is {BLOCK} tokens and its <em>target</em> is
            the same window shifted one step right — so the model is always asked &ldquo;given these
            tokens, predict the next one.&rdquo; Your text currently makes {dsCount} windows.
          </p>
          <p>
            <strong>Randomization?</strong> None. This stage is fully deterministic — the vocabulary
            and windows are a pure function of the text. No weights, no randomness yet.
          </p>
          <p>
            Details: <Link href="/learn/text-into-tokens">Text into Tokens</Link> ·{" "}
            <Link href="/learn/vocabulary-token-ids">The Vocabulary &amp; Token IDs</Link>.
          </p>
        </Disclosure>
      </div>

      {/* 2 · Weights */}
      <div className="card card-pad">
        <div className="card-title">2 · Weights (token embeddings) — click a cell to edit</div>
        {ready && p ? (
          <>
            <div style={{ overflow: "auto", maxHeight: 380, border: "1px solid var(--border)", borderRadius: 10 }}>
              <table style={{ borderCollapse: "collapse", fontSize: "0.8rem" }}>
                <thead>
                  <tr>
                    <th style={{ position: "sticky", left: 0, background: "var(--surface-2)", padding: "4px 8px", textAlign: "left" }}>
                      token
                    </th>
                    {Array.from({ length: embedDim }, (_, d) => (
                      <th key={d} className="mono" style={{ padding: "4px 6px", color: "var(--text-3)", fontWeight: 500 }}>
                        dim {d}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {vocab.map((t, i) => (
                    <tr key={t.id}>
                      <th style={{ position: "sticky", left: 0, background: "var(--surface-2)", padding: "2px 8px", textAlign: "left" }}>
                        <span className="mono" style={{ color: tokenColor(t.id), fontWeight: 700 }}>
                          {t.display}
                        </span>
                      </th>
                      {Array.from({ length: embedDim }, (_, j) => (
                        <td key={j} style={{ padding: 1 }}>
                          <input
                            key={`${i}-${j}-${version}`}
                            type="number"
                            step="0.01"
                            defaultValue={p.wte[i][j].toFixed(3)}
                            disabled={training}
                            onBlur={(e) => commitCell(i, j, e.target.value)}
                            style={{
                              width: 62,
                              padding: "3px 4px",
                              fontFamily: "var(--font-mono)",
                              fontSize: "0.8rem",
                              border: "1px solid transparent",
                              borderRadius: 5,
                              background: "var(--surface-2)",
                              color: "var(--text)",
                            }}
                          />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="vis-controls">
              <button className="btn btn-ghost btn-sm" onClick={randomizeEmbeddings} disabled={training}>
                Randomize embeddings
              </button>
              <button className="btn btn-ghost btn-sm" onClick={resetAll} disabled={training}>
                Reset all weights
              </button>
              <button className="btn btn-ghost btn-sm" onClick={downloadModel} disabled={training}>
                ⬇ Download model
              </button>
              <button className="btn btn-ghost btn-sm" onClick={() => modelFileRef.current?.click()} disabled={training}>
                ⬆ Upload model
              </button>
              <input ref={modelFileRef} type="file" accept=".json,application/json" hidden onChange={onModelFile} />
            </div>

            <Disclosure title="What's happening here?">
              <p>
                <strong>These are the model&apos;s parameters.</strong> The grid is the
                token-embedding matrix — one row per token, one number per embedding dimension. Each
                row is that token&apos;s starting &ldquo;meaning vector.&rdquo;
              </p>
              <p>
                <strong>They start random.</strong> Weights begin as small Gaussian noise (scaled ×
                0.08). On first load they use a fixed seed (so the demo is reproducible);{" "}
                <em>Reset all weights</em> re-rolls everything with a fresh random seed, and{" "}
                <em>Randomize embeddings</em> re-rolls only the token table.
              </p>
              <p>
                <strong>You can edit them.</strong> Click any cell to hand-set a number — training
                then works from whatever you put here.
              </p>
              <p>
                <strong>Training moves them.</strong> Every step nudges these numbers (and the rest
                of the network) to lower the loss, so similar tokens drift toward similar
                vectors.
              </p>
              <p>
                <strong>Randomization?</strong> Yes — randomize/reset re-roll weights at random,
                while the initial fixed-seed state is deterministic. Editing is manual, not random.
              </p>
              <p>
                Details: <Link href="/learn/embeddings">Embeddings: Words as Vectors</Link>.
              </p>
            </Disclosure>
          </>
        ) : (
          <p className="small muted">Preparing model…</p>
        )}
      </div>

      {/* 3 · Train */}
      <div className="card card-pad">
        <div className="card-title">3 · Train step by step</div>
        {ready && dsCount > 0 ? (
          <>
            <div className="vis-controls">
              <label>
                Loss function:{" "}
                <select
                  value={lossFn}
                  onChange={(e) => changeLossFn(e.target.value as LossFn)}
                  disabled={training}
                  style={{
                    fontFamily: "var(--font-sans)",
                    padding: "6px 8px",
                    borderRadius: 8,
                    border: "1px solid var(--border-strong)",
                    background: "var(--surface)",
                    color: "var(--text)",
                  }}
                >
                  {LOSS_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {training ? (
              <div className="row" style={{ gap: 12, marginTop: 12 }}>
                <span className="spinner" />
                <span className="small muted">
                  training… {trainProg ? `${trainProg.done} / ${trainProg.total}` : ""}
                </span>
                <span style={{ flex: 1 }}>
                  <span className="progress-bar" style={{ display: "block" }}>
                    <span
                      className="progress-fill"
                      style={{ width: `${trainProg ? (trainProg.done / trainProg.total) * 100 : 0}%` }}
                    />
                  </span>
                </span>
                <button className="btn btn-ghost btn-sm" onClick={() => (cancelRef.current = true)}>
                  Stop
                </button>
              </div>
            ) : (
              <div className="vis-controls">
                <button className="btn btn-primary btn-sm" onClick={() => runTraining(1)}>
                  1 step
                </button>
                <button className="btn btn-primary btn-sm" onClick={() => runTraining(10)}>
                  10 steps
                </button>
                <button className="btn btn-primary btn-sm" onClick={() => runTraining(200)}>
                  200 steps
                </button>
                <span className="row" style={{ gap: 6 }}>
                  <input
                    type="number"
                    min={1}
                    max={100000}
                    value={stepsInput}
                    onChange={(e) => setStepsInput(Math.max(1, Number(e.target.value) || 1))}
                    style={{
                      width: 80,
                      padding: "5px 8px",
                      fontFamily: "var(--font-mono)",
                      borderRadius: 8,
                      border: "1px solid var(--border-strong)",
                      background: "var(--surface)",
                      color: "var(--text)",
                    }}
                  />
                  <button className="btn btn-accent-soft btn-sm" onClick={() => runTraining(stepsInput)}>
                    Train
                  </button>
                </span>
                <span className="small muted">
                  step <strong>{step}</strong> · loss{" "}
                  <strong>{loss != null ? loss.toFixed(4) : "—"}</strong>
                </span>
              </div>
            )}

            {history.length >= 2 && (
              <svg viewBox="0 0 300 60" style={{ width: "100%", marginTop: 10 }}>
                <polyline points={spark} fill="none" stroke="var(--accent)" strokeWidth={2} />
              </svg>
            )}
            <p className="small faint" style={{ marginTop: 8, marginBottom: 0 }}>
              Each step: forward → loss → backward → Adam update, on a mini-batch of 4 windows.
              With cross-entropy, the loss starts near <span className="mono">ln({vocab.length}) ≈ {Math.log(vocab.length).toFixed(1)}</span>{" "}
              and falls as the model learns your text.
            </p>

            <Disclosure title="What's happening here?">
              <p>
                <strong>One step = forward → loss → backward → update.</strong> The forward pass
                computes a probability for every possible next token; the loss measures how wrong
                the guess is; backprop runs the chain rule backward to compute the gradient of the
                loss with respect to <em>every</em> weight; then Adam nudges each weight to reduce
                the loss.
              </p>
              <p>
                <strong>The math.</strong> Loss is cross-entropy (default) or mean-squared error.
                Gradients are averaged over a mini-batch of 4 windows, globally clipped to norm 1.0,
                then applied with Adam (learning rate 0.02, β₁ = 0.9, β₂ = 0.999) plus weight decay
                0.01.
              </p>
              <p>
                <strong>Randomization?</strong> No — training is deterministic. Given your text and
                the current weights, each step&apos;s loss and update are exactly determined (windows
                are visited in a fixed cyclic order). The only randomness in the playground lives in
                initialization (step 2) and generation (step 4).
              </p>
              <p>
                Details: <Link href="/learn/loss">Loss</Link> ·{" "}
                <Link href="/learn/gradient-descent">Gradient Descent</Link> ·{" "}
                <Link href="/learn/backpropagation">Backpropagation</Link> ·{" "}
                <Link href="/learn/optimizer">The Optimizer</Link> ·{" "}
                <Link href="/learn/training-loop">The Training Loop</Link>.
              </p>
            </Disclosure>
          </>
        ) : (
          <p className="small muted">Add at least {BLOCK + 1} tokens of text to train.</p>
        )}
      </div>

      {/* 4 · Predict */}
      <div className="card card-pad">
        <div className="card-title">4 · Predict the next characters &amp; words</div>
        {ready && dsCount > 0 ? (
          <>
            <div className="row" style={{ gap: 8 }}>
              <input
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="start typing…"
                style={{
                  flex: 1,
                  fontFamily: "var(--font-mono)",
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid var(--border-strong)",
                  background: "var(--surface-2)",
                  color: "var(--text)",
                }}
              />
              <button className="btn btn-primary" onClick={predict}>
                Predict next
              </button>
              <button className="btn btn-ghost" onClick={generateText}>
                Generate 40 tokens
              </button>
              <label className="small">
                temp
                <input type="range" min={0.1} max={1.8} step={0.1} value={temp} onChange={(e) => setTemp(Number(e.target.value))} />
              </label>
            </div>

            {unknownInPrompt > 0 && (
              <p className="small" style={{ color: "var(--warn)", marginTop: 8 }}>
                {unknownInPrompt} token(s) in your prompt aren&apos;t in the vocabulary and were
                skipped.
              </p>
            )}

            {predictions.length > 0 && (
              <div className="stack" style={{ gap: 3, marginTop: 12 }}>
                {predictions.map((r) => (
                  <div key={r.id} className="row" style={{ gap: 8, alignItems: "center" }}>
                    <span className="mono" style={{ minWidth: 24, color: tokenColor(r.id), fontWeight: 700 }}>
                      {vocab[r.id].display}
                    </span>
                    <span style={{ flex: 1, height: 12, background: "var(--surface-3)", borderRadius: 4, overflow: "hidden" }}>
                      <span
                        style={{
                          display: "block",
                          height: "100%",
                          background: "var(--accent)",
                          opacity: 0.35 + 0.65 * (r.p / predictions[0].p),
                          width: `${(r.p / predictions[0].p) * 100}%`,
                          borderRadius: 4,
                        }}
                      />
                    </span>
                    <span className="mono small" style={{ width: 40, textAlign: "right" }}>
                      {(r.p * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            {generatedIds.length > 0 && (
              <div
                style={{
                  marginTop: 14,
                  padding: 14,
                  background: "var(--surface-2)",
                  border: "1px solid var(--border)",
                  borderRadius: 10,
                  fontFamily: "var(--font-mono)",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                  lineHeight: 2,
                }}
              >
                <span style={{ color: "var(--text-3)" }}>{prompt}</span>
                {generatedIds.map((id, i) => {
                  const tok = vocab[id];
                  return (
                    <span key={i} style={{ color: tokenColor(id) }}>
                      {(tokenizer === "word" ? " " : "") +
                        (tok.char === "\n" ? "⏎\n" : tok.char)}
                    </span>
                  );
                })}
              </div>
            )}

            <Disclosure title="What's happening here?">
              <p>
                <strong>Predict next.</strong> A forward pass runs over your prompt (its last {BLOCK}{" "}
                tokens), then the final position&apos;s logits are softmax-ed into a probability
                distribution. The bars show the top 6 likely next tokens and their
                probabilities.
              </p>
              <p>
                <strong>Generate.</strong> Instead of always taking the top choice, it{" "}
                <em>samples</em> from the distribution, appends the result, and repeats 40 times
                (autoregressive).
              </p>
              <p>
                <strong>Temperature</strong> reshapes the distribution: low ≈ greedy/confident
                (almost always the top token), high ≈ flat (more varied, less predictable).
              </p>
              <p>
                <strong>Randomization?</strong> Partly. Prediction (softmax) is deterministic.
                Generation is random — each sample draws from the distribution with a fresh seed, so
                runs differ.
              </p>
              <p>
                Details: <Link href="/learn/logits-softmax">Logits &amp; Softmax</Link> ·{" "}
                <Link href="/learn/generate-text">Generating Text</Link>.
              </p>
            </Disclosure>
          </>
        ) : (
          <p className="small muted">Train a little first (add text, then train).</p>
        )}
      </div>

      {notice && (
        <div className="callout callout-tip">
          <p style={{ margin: 0 }} className="small">
            {notice}
          </p>
        </div>
      )}
    </div>
  );
}
