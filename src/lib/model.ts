// ToyGPT — a genuinely-trainable, configurable, tiny character-level transformer.
// Everything runs in the browser (or Node) with no external deps.

import type { ModelConfig } from "./corpus";

// --- Parameters -------------------------------------------------------------

export interface Params {
  cfg: ModelConfig;
  wte: number[][]; // [vocabSize][embedDim] token embedding
  wpe: number[][]; // [blockSize][embedDim] position embedding
  attn: {
    wq: number[][]; // [embedDim][embedDim]
    wk: number[][];
    wv: number[][];
    wo: number[][];
    bq: number[];
    bk: number[];
    bv: number[];
    bo: number[];
  };
  ln1w: number[];
  ln1b: number[];
  mlp: { w1: number[][]; b1: number[]; w2: number[][]; b2: number[] }; // w1 [E][4E], w2 [4E][E]
  ln2w: number[];
  ln2b: number[];
  lnfw: number[];
  lnfb: number[];
  lm: { w: number[][]; b: number[] }; // w [E][V]
}

function zeros(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}
function zerosVec(n: number): number[] {
  return new Array<number>(n).fill(0);
}
function onesVec(n: number): number[] {
  return new Array<number>(n).fill(1);
}

/** Deterministic PRNG so training / generation are reproducible. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randNormal(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function randMat(rows: number, cols: number, rng: () => number, scale: number): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => randNormal(rng) * scale),
  );
}

/** Initialize parameters for a given config with a fixed seed. */
export function initParams(cfg: ModelConfig, seed = 42): Params {
  const rng = mulberry32(seed);
  const E = cfg.embedDim;
  const V = cfg.vocabSize;
  const M = E * 4;
  return {
    cfg,
    wte: randMat(V, E, rng, 0.08),
    wpe: randMat(cfg.blockSize, E, rng, 0.08),
    attn: {
      wq: randMat(E, E, rng, 0.05),
      wk: randMat(E, E, rng, 0.05),
      wv: randMat(E, E, rng, 0.05),
      wo: randMat(E, E, rng, 0.05),
      bq: zerosVec(E),
      bk: zerosVec(E),
      bv: zerosVec(E),
      bo: zerosVec(E),
    },
    ln1w: onesVec(E),
    ln1b: zerosVec(E),
    mlp: {
      w1: randMat(E, M, rng, 0.05),
      b1: zerosVec(M),
      w2: randMat(M, E, rng, 0.05),
      b2: zerosVec(E),
    },
    ln2w: onesVec(E),
    ln2b: zerosVec(E),
    lnfw: onesVec(E),
    lnfb: zerosVec(E),
    lm: { w: randMat(E, V, rng, 0.05), b: zerosVec(V) },
  };
}

/** All-zero parameter structure with the same shapes (for gradients / Adam). */
export function zeroParams(cfg: ModelConfig): Params {
  const E = cfg.embedDim;
  const V = cfg.vocabSize;
  const M = E * 4;
  return {
    cfg,
    wte: zeros(V, E),
    wpe: zeros(cfg.blockSize, E),
    attn: {
      wq: zeros(E, E),
      wk: zeros(E, E),
      wv: zeros(E, E),
      wo: zeros(E, E),
      bq: zerosVec(E),
      bk: zerosVec(E),
      bv: zerosVec(E),
      bo: zerosVec(E),
    },
    ln1w: zerosVec(E),
    ln1b: zerosVec(E),
    mlp: { w1: zeros(E, M), b1: zerosVec(M), w2: zeros(M, E), b2: zerosVec(E) },
    ln2w: zerosVec(E),
    ln2b: zerosVec(E),
    lnfw: zerosVec(E),
    lnfb: zerosVec(E),
    lm: { w: zeros(E, V), b: zerosVec(V) },
  };
}

/** Flatten every parameter (matrices row-major, vectors) in a fixed order. */
export function collectArrays(p: Params): number[][] {
  return [
    ...p.wte,
    ...p.wpe,
    ...p.attn.wq,
    ...p.attn.wk,
    ...p.attn.wv,
    ...p.attn.wo,
    p.attn.bq,
    p.attn.bk,
    p.attn.bv,
    p.attn.bo,
    p.ln1w,
    p.ln1b,
    ...p.mlp.w1,
    p.mlp.b1,
    ...p.mlp.w2,
    p.mlp.b2,
    p.ln2w,
    p.ln2b,
    p.lnfw,
    p.lnfb,
    ...p.lm.w,
    p.lm.b,
  ];
}

export function countParams(p: Params): number {
  let n = 0;
  for (const arr of collectArrays(p)) n += arr.length;
  return n;
}

// --- Math helpers -----------------------------------------------------------

export function matmul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const k = B.length;
  const n = B[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
      C[i][j] = s;
    }
  }
  return C;
}

function add(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

export function softmaxRow(z: number[]): number[] {
  const max = Math.max(...z);
  const exps = z.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function softmaxRows(z: number[][]): number[][] {
  return z.map(softmaxRow);
}

export function gelu(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}

export function dGelu(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  const y = x + 0.044715 * x * x * x;
  const t = Math.tanh(c * y);
  return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x);
}

export function crossEntropy(probs: number[][], target: number[]): number {
  let sum = 0;
  const T = probs.length;
  for (let t = 0; t < T; t++) sum += -Math.log(Math.max(probs[t][target[t]], 1e-12));
  return sum / T;
}

// --- Loss functions (selectable in the playground) --------------------------

export type LossFn = "ce" | "mse";

/** Scalar loss value (averaged over positions) for a given loss function. */
export function lossValue(fn: LossFn, probs: number[][], target: number[]): number {
  const T = probs.length;
  let sum = 0;
  if (fn === "ce") {
    for (let t = 0; t < T; t++) sum += -Math.log(Math.max(probs[t][target[t]], 1e-12));
    return sum / T;
  }
  // mean squared error (½ Σ (p - y)², averaged over positions)
  for (let t = 0; t < T; t++) {
    for (let j = 0; j < probs[t].length; j++) {
      const y = j === target[t] ? 1 : 0;
      const d = probs[t][j] - y;
      sum += 0.5 * d * d;
    }
  }
  return sum / T;
}

/** Gradient of the loss w.r.t. the logits (shape [T][V]). */
export function lossGradLogits(fn: LossFn, probs: number[][], target: number[]): number[][] {
  const T = probs.length;
  const V = probs[0].length;
  const out = probs.map(() => new Array<number>(V).fill(0));
  for (let t = 0; t < T; t++) {
    const p = probs[t];
    if (fn === "ce") {
      for (let j = 0; j < V; j++) out[t][j] = (p[j] - (j === target[t] ? 1 : 0)) / T;
    } else {
      let S = 0;
      for (let j = 0; j < V; j++) {
        const y = j === target[t] ? 1 : 0;
        S += (p[j] - y) * p[j];
      }
      for (let i = 0; i < V; i++) {
        const y = i === target[t] ? 1 : 0;
        out[t][i] = (p[i] * (p[i] - y - S)) / T;
      }
    }
  }
  return out;
}

// --- Forward pass -----------------------------------------------------------

interface LNState {
  out: number[][];
  xn: number[][];
  mean: number[];
  rstd: number[];
  gamma: number[];
  beta: number[];
}

interface AttnState {
  x: number[][]; // input to attention (= ln1.out)
  q: number[][][]; // [H][T][HD]
  k: number[][][];
  v: number[][][];
  scores: number[][][]; // [H][T][T] (causal)
  probs: number[][][];
  cat: number[][]; // concatenated heads, before output projection
  a1: number[][]; // attention output
}

interface MLPState {
  pre: number[][];
  act: number[][];
  out: number[][];
}

export interface ForwardCache {
  T: number;
  idx: number[];
  x: number[][]; // embeddings (wte + wpe)
  ln1: LNState;
  attn: AttnState;
  h1: number[][];
  ln2: LNState;
  mlp: MLPState;
  h2: number[][];
  lnf: LNState;
  logits: number[][];
  probs: number[][];
}

function layernorm(x: number[][], gamma: number[], beta: number[]): LNState {
  const T = x.length;
  const E = x[0].length;
  const out: number[][] = [];
  const xn: number[][] = [];
  const mean: number[] = [];
  const rstd: number[] = [];
  for (let t = 0; t < T; t++) {
    let m = 0;
    for (let e = 0; e < E; e++) m += x[t][e];
    m /= E;
    let v = 0;
    for (let e = 0; e < E; e++) {
      const d = x[t][e] - m;
      v += d * d;
    }
    v /= E;
    const r = 1 / Math.sqrt(v + 1e-5);
    const nrm = new Array<number>(E);
    const o = new Array<number>(E);
    for (let e = 0; e < E; e++) {
      nrm[e] = (x[t][e] - m) * r;
      o[e] = nrm[e] * gamma[e] + beta[e];
    }
    mean.push(m);
    rstd.push(r);
    xn.push(nrm);
    out.push(o);
  }
  return { out, xn, mean, rstd, gamma, beta };
}

function attention(p: Params, x: number[][]): AttnState {
  const T = x.length;
  const E = p.cfg.embedDim;
  const H = p.cfg.nHead;
  const HD = E / H;
  const q = matmul(x, p.attn.wq);
  const k = matmul(x, p.attn.wk);
  const v = matmul(x, p.attn.wv);
  for (let t = 0; t < T; t++)
    for (let e = 0; e < E; e++) {
      q[t][e] += p.attn.bq[e];
      k[t][e] += p.attn.bk[e];
      v[t][e] += p.attn.bv[e];
    }

  const scale = 1 / Math.sqrt(HD);
  const qh: number[][][] = [];
  const kh: number[][][] = [];
  const vh: number[][][] = [];
  const scores: number[][][] = [];
  const probs: number[][][] = [];
  const headsOut: number[][][] = [];

  for (let h = 0; h < H; h++) {
    const off = h * HD;
    const qm = q.map((r) => r.slice(off, off + HD));
    const km = k.map((r) => r.slice(off, off + HD));
    const vm = v.map((r) => r.slice(off, off + HD));

    const sc: number[][] = [];
    for (let t = 0; t < T; t++) {
      const row = new Array<number>(T).fill(-Infinity);
      for (let s = 0; s <= t; s++) {
        let d = 0;
        for (let e = 0; e < HD; e++) d += qm[t][e] * km[s][e];
        row[s] = d * scale;
      }
      sc.push(row);
    }
    const pr = softmaxRows(sc);
    const om: number[][] = [];
    for (let t = 0; t < T; t++) {
      const row = new Array<number>(HD).fill(0);
      for (let s = 0; s <= t; s++) {
        const w = pr[t][s];
        for (let e = 0; e < HD; e++) row[e] += w * vm[s][e];
      }
      om.push(row);
    }
    qh.push(qm);
    kh.push(km);
    vh.push(vm);
    scores.push(sc);
    probs.push(pr);
    headsOut.push(om);
  }

  const cat: number[][] = [];
  for (let t = 0; t < T; t++) {
    const row = new Array<number>(E);
    for (let h = 0; h < H; h++)
      for (let e = 0; e < HD; e++) row[h * HD + e] = headsOut[h][t][e];
    cat.push(row);
  }
  const a1 = matmul(cat, p.attn.wo).map((r, i) => r.map((v, j) => v + p.attn.bo[j]));

  return { x, q: qh, k: kh, v: vh, scores, probs, cat, a1 };
}

function mlpForward(p: Params, x: number[][]): MLPState {
  const pre = matmul(x, p.mlp.w1).map((r, i) => r.map((v, j) => v + p.mlp.b1[j]));
  const act = pre.map((row) => row.map(gelu));
  const out = matmul(act, p.mlp.w2).map((r, i) => r.map((v, j) => v + p.mlp.b2[j]));
  return { pre, act, out };
}

export function forward(p: Params, idx: number[]): ForwardCache {
  const T = idx.length;
  const E = p.cfg.embedDim;
  const x: number[][] = [];
  for (let t = 0; t < T; t++) {
    const row = new Array<number>(E);
    const te = p.wte[idx[t]];
    const pe = p.wpe[t];
    for (let e = 0; e < E; e++) row[e] = te[e] + pe[e];
    x.push(row);
  }
  const ln1 = layernorm(x, p.ln1w, p.ln1b);
  const attn = attention(p, ln1.out);
  const h1 = add(x, attn.a1);
  const ln2 = layernorm(h1, p.ln2w, p.ln2b);
  const mlp = mlpForward(p, ln2.out);
  const h2 = add(h1, mlp.out);
  const lnf = layernorm(h2, p.lnfw, p.lnfb);
  const logits = matmul(lnf.out, p.lm.w).map((r, i) => r.map((v, j) => v + p.lm.b[j]));
  const probs = softmaxRows(logits);
  return { T, idx, x, ln1, attn, h1, ln2, mlp, h2, lnf, logits, probs };
}

// --- Sampling / generation --------------------------------------------------

export function sampleNext(
  logits: number[],
  temperature: number,
  topK: number,
  rng: () => number,
): number {
  const z = logits.map((l) => l / Math.max(temperature, 1e-6));
  const max = Math.max(...z);
  const exps = z.map((v) => Math.exp(v - max));
  let sum = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((v) => v / sum);

  if (topK > 0 && topK < probs.length) {
    const order = probs
      .map((p, i) => i)
      .sort((a, b) => probs[b] - probs[a]);
    const allowed = new Set(order.slice(0, topK));
    for (let i = 0; i < probs.length; i++) if (!allowed.has(i)) probs[i] = 0;
    sum = probs.reduce((a, b) => a + b, 0);
    for (let i = 0; i < probs.length; i++) probs[i] /= sum;
  }

  const r = rng();
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (r <= acc) return i;
  }
  return probs.length - 1;
}

/** Generate `maxTokens` new tokens, continuing from `prompt` (token ids). */
export function generate(
  p: Params,
  prompt: number[],
  maxTokens: number,
  temperature: number,
  topK: number,
  rng: () => number,
): number[] {
  const out = prompt.slice();
  for (let i = 0; i < maxTokens; i++) {
    const ctx = out.slice(-p.cfg.blockSize);
    const cache = forward(p, ctx);
    const logits = cache.logits[cache.logits.length - 1];
    out.push(sampleNext(logits, temperature, topK, rng));
  }
  return out;
}
