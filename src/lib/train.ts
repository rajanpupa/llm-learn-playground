// Backpropagation, the Adam optimizer, and the training step for ToyGPT.

import type { Example, ModelConfig } from "./corpus";
import {
  type Params,
  type ForwardCache,
  forward,
  zeroParams,
  collectArrays,
  dGelu,
  lossGradLogits,
  lossValue,
  type LossFn,
} from "./model";

// --- Small local helpers ----------------------------------------------------

function zeros(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(0));
}

// --- Backward pass ----------------------------------------------------------

function layernormBackward(
  ln: ForwardCache["ln1"],
  dOut: number[][],
  gw: number[],
  gb: number[],
): number[][] {
  const T = dOut.length;
  const E = dOut[0].length;
  const gamma = ln.gamma;
  const dx: number[][] = [];
  for (let t = 0; t < T; t++) {
    const xn = ln.xn[t];
    const rstd = ln.rstd[t];
    const dout = dOut[t];
    for (let e = 0; e < E; e++) {
      gw[e] += dout[e] * xn[e];
      gb[e] += dout[e];
    }
    const dxn = new Array<number>(E);
    let meanDxn = 0;
    for (let e = 0; e < E; e++) {
      dxn[e] = dout[e] * gamma[e];
      meanDxn += dxn[e];
    }
    meanDxn /= E;
    let meanDxnXn = 0;
    for (let e = 0; e < E; e++) meanDxnXn += dxn[e] * xn[e];
    meanDxnXn /= E;
    const row = new Array<number>(E);
    for (let e = 0; e < E; e++) row[e] = (dxn[e] - meanDxn - xn[e] * meanDxnXn) * rstd;
    dx.push(row);
  }
  return dx;
}

function attentionBackward(p: Params, attn: ForwardCache["attn"], dA1: number[][], g: Params): number[][] {
  const T = dA1.length;
  const E = p.cfg.embedDim;
  const H = p.cfg.nHead;
  const HD = E / H;
  // d_cat = dA1 @ wo^T
  const dCat: number[][] = zeros(T, E);
  for (let t = 0; t < T; t++)
    for (let e = 0; e < E; e++) {
      let s = 0;
      for (let f = 0; f < E; f++) s += dA1[t][f] * p.attn.wo[e][f];
      dCat[t][e] = s;
    }
  // d_wo = cat^T @ dA1 ; d_bo = sum dA1
  for (let e = 0; e < E; e++)
    for (let f = 0; f < E; f++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += attn.cat[t][e] * dA1[t][f];
      g.attn.wo[e][f] += s;
    }
  for (let f = 0; f < E; f++) {
    let s = 0;
    for (let t = 0; t < T; t++) s += dA1[t][f];
    g.attn.bo[f] += s;
  }

  const dQ = zeros(T, E);
  const dK = zeros(T, E);
  const dV = zeros(T, E);
  const scale = 1 / Math.sqrt(HD);

  for (let h = 0; h < H; h++) {
    const qh = attn.q[h];
    const kh = attn.k[h];
    const vh = attn.v[h];
    const pr = attn.probs[h];
    const off = h * HD;

    const dOutH = dCat.map((r) => r.slice(off, off + HD));

    const dVh = zeros(T, HD);
    for (let s = 0; s < T; s++)
      for (let t = s; t < T; t++) {
        const w = pr[t][s];
        for (let e = 0; e < HD; e++) dVh[s][e] += w * dOutH[t][e];
      }

    const dPr = zeros(T, T);
    for (let t = 0; t < T; t++)
      for (let s = 0; s <= t; s++) {
        let d = 0;
        for (let e = 0; e < HD; e++) d += dOutH[t][e] * vh[s][e];
        dPr[t][s] = d;
      }

    const dScores = zeros(T, T);
    for (let t = 0; t < T; t++) {
      let c = 0;
      for (let j = 0; j <= t; j++) c += pr[t][j] * dPr[t][j];
      for (let s = 0; s <= t; s++) dScores[t][s] = pr[t][s] * (dPr[t][s] - c);
    }

    const dQh = zeros(T, HD);
    const dKh = zeros(T, HD);
    for (let t = 0; t < T; t++)
      for (let s = 0; s <= t; s++) {
        const gs = dScores[t][s] * scale;
        for (let e = 0; e < HD; e++) {
          dQh[t][e] += gs * kh[s][e];
          dKh[s][e] += gs * qh[t][e];
        }
      }

    for (let t = 0; t < T; t++)
      for (let e = 0; e < HD; e++) {
        dQ[t][off + e] += dQh[t][e];
        dK[t][off + e] += dKh[t][e];
        dV[t][off + e] += dVh[t][e];
      }
  }

  const x = attn.x;
  const projectBack = (d: number[][], w: number[][], gw: number[][], gb: number[]): number[][] => {
    for (let e = 0; e < E; e++)
      for (let f = 0; f < E; f++) {
        let s = 0;
        for (let t = 0; t < T; t++) s += x[t][e] * d[t][f];
        gw[e][f] += s;
      }
    for (let f = 0; f < E; f++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += d[t][f];
      gb[f] += s;
    }
    const dX = zeros(T, E);
    for (let t = 0; t < T; t++)
      for (let e = 0; e < E; e++) {
        let s = 0;
        for (let f = 0; f < E; f++) s += d[t][f] * w[e][f];
        dX[t][e] = s;
      }
    return dX;
  };

  const dx = zeros(T, E);
  const dxQ = projectBack(dQ, p.attn.wq, g.attn.wq, g.attn.bq);
  const dxK = projectBack(dK, p.attn.wk, g.attn.wk, g.attn.bk);
  const dxV = projectBack(dV, p.attn.wv, g.attn.wv, g.attn.bv);
  for (let t = 0; t < T; t++)
    for (let e = 0; e < E; e++) dx[t][e] = dxQ[t][e] + dxK[t][e] + dxV[t][e];
  return dx;
}

function mlpBackward(
  p: Params,
  mlp: ForwardCache["mlp"],
  x: number[][],
  dOut: number[][],
  g: Params,
): number[][] {
  const T = dOut.length;
  const E = p.cfg.embedDim;
  const M = p.mlp.w1[0].length;
  for (let i = 0; i < M; i++)
    for (let e = 0; e < E; e++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += mlp.act[t][i] * dOut[t][e];
      g.mlp.w2[i][e] += s;
    }
  for (let e = 0; e < E; e++) {
    let s = 0;
    for (let t = 0; t < T; t++) s += dOut[t][e];
    g.mlp.b2[e] += s;
  }
  const dAct = zeros(T, M);
  for (let t = 0; t < T; t++)
    for (let i = 0; i < M; i++) {
      let s = 0;
      for (let e = 0; e < E; e++) s += dOut[t][e] * p.mlp.w2[i][e];
      dAct[t][i] = s;
    }
  const dPre = dAct.map((row, t) => row.map((v, i) => v * dGelu(mlp.pre[t][i])));
  for (let e = 0; e < E; e++)
    for (let i = 0; i < M; i++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += x[t][e] * dPre[t][i];
      g.mlp.w1[e][i] += s;
    }
  for (let i = 0; i < M; i++) {
    let s = 0;
    for (let t = 0; t < T; t++) s += dPre[t][i];
    g.mlp.b1[i] += s;
  }
  const dX = zeros(T, E);
  for (let t = 0; t < T; t++)
    for (let e = 0; e < E; e++) {
      let s = 0;
      for (let i = 0; i < M; i++) s += dPre[t][i] * p.mlp.w1[e][i];
      dX[t][e] = s;
    }
  return dX;
}

/** Accumulate gradients of cross-entropy loss into `g`. */
export function backward(
  cache: ForwardCache,
  p: Params,
  target: number[],
  g: Params,
  lossFn: LossFn = "ce",
): void {
  const T = cache.T;
  const E = p.cfg.embedDim;
  const V = p.cfg.vocabSize;
  // gradient of the chosen loss w.r.t. the logits
  const dLogits = lossGradLogits(lossFn, cache.probs, target);

  // LM head
  for (let t = 0; t < T; t++)
    for (let j = 0; j < V; j++) g.lm.b[j] += dLogits[t][j];
  for (let e = 0; e < E; e++)
    for (let j = 0; j < V; j++) {
      let s = 0;
      for (let t = 0; t < T; t++) s += cache.lnf.out[t][e] * dLogits[t][j];
      g.lm.w[e][j] += s;
    }
  const dLnfOut = zeros(T, E);
  for (let t = 0; t < T; t++)
    for (let e = 0; e < E; e++) {
      let s = 0;
      for (let j = 0; j < V; j++) s += dLogits[t][j] * p.lm.w[e][j];
      dLnfOut[t][e] = s;
    }

  // LNf
  const dH2 = layernormBackward(cache.lnf, dLnfOut, g.lnfw, g.lnfb);

  // h2 = h1 + mlp.out
  const dH1 = dH2.map((r) => r.slice());
  const dMlpOut = dH2;

  // MLP
  const dLn2Out = mlpBackward(p, cache.mlp, cache.ln2.out, dMlpOut, g);

  // LN2
  const dH1Ln = layernormBackward(cache.ln2, dLn2Out, g.ln2w, g.ln2b);
  for (let t = 0; t < T; t++)
    for (let e = 0; e < E; e++) dH1[t][e] += dH1Ln[t][e];

  // h1 = x + a1
  const dX = dH1.map((r) => r.slice());
  const dA1 = dH1;

  // attention
  const dLn1Out = attentionBackward(p, cache.attn, dA1, g);

  // LN1
  const dXLn = layernormBackward(cache.ln1, dLn1Out, g.ln1w, g.ln1b);
  for (let t = 0; t < T; t++)
    for (let e = 0; e < E; e++) dX[t][e] += dXLn[t][e];

  // embeddings
  for (let t = 0; t < T; t++)
    for (let e = 0; e < E; e++) {
      g.wte[cache.idx[t]][e] += dX[t][e];
      g.wpe[t][e] += dX[t][e];
    }
}

// --- Optimizer (AdamW) ------------------------------------------------------

export interface AdamConfig {
  lr: number;
  b1: number;
  b2: number;
  eps: number;
  wd: number;
}

export const DEFAULT_ADAM: AdamConfig = { lr: 0.02, b1: 0.9, b2: 0.999, eps: 1e-8, wd: 0.01 };

export interface AdamState {
  m: Params;
  v: Params;
  step: number;
}

export function initAdam(cfg: ModelConfig): AdamState {
  return { m: zeroParams(cfg), v: zeroParams(cfg), step: 0 };
}

/** Global L2 norm of the gradient (used for clipping / the lesson 13 viz). */
export function gradNorm(g: Params): number {
  let s = 0;
  for (const arr of collectArrays(g)) for (const x of arr) s += x * x;
  return Math.sqrt(s);
}

/** Clip gradients by global norm. Returns the factor applied (1 if no clip). */
export function clipGrad(g: Params, maxNorm: number): number {
  const n = gradNorm(g);
  if (n <= maxNorm || n === 0) return 1;
  const f = maxNorm / n;
  for (const arr of collectArrays(g)) for (let i = 0; i < arr.length; i++) arr[i] *= f;
  return f;
}

export function scaleGrads(g: Params, s: number): void {
  for (const arr of collectArrays(g)) for (let i = 0; i < arr.length; i++) arr[i] *= s;
}

function adamUpdate(p: Params, g: Params, m: Params, v: Params, cfg: AdamConfig, step: number): void {
  const b1 = cfg.b1;
  const b2 = cfg.b2;
  const bc1 = 1 - Math.pow(b1, step);
  const bc2 = 1 - Math.pow(b2, step);
  const lr = cfg.lr;
  const eps = cfg.eps;
  const wd = cfg.wd;

  const ps = collectArrays(p);
  const gs = collectArrays(g);
  const ms = collectArrays(m);
  const vs = collectArrays(v);

  for (let i = 0; i < ps.length; i++) {
    const pr = ps[i];
    const gr = gs[i];
    const mr = ms[i];
    const vr = vs[i];
    for (let j = 0; j < pr.length; j++) {
      const gi = gr[j];
      mr[j] = b1 * mr[j] + (1 - b1) * gi;
      vr[j] = b2 * vr[j] + (1 - b2) * gi * gi;
      const mhat = mr[j] / bc1;
      const vhat = vr[j] / bc2;
      pr[j] -= lr * (mhat / (Math.sqrt(vhat) + eps) + wd * pr[j]);
    }
  }
}

// --- Training step ----------------------------------------------------------

export interface TrainConfig {
  adam: AdamConfig;
  clip: number; // global gradient-clip norm (0 = disabled)
}

export const DEFAULT_TRAIN: TrainConfig = { adam: { ...DEFAULT_ADAM }, clip: 1.0 };

/** One optimizer step over a mini-batch. Returns the average loss of the batch. */
export function trainStep(
  p: Params,
  adam: AdamState,
  batch: Example[],
  cfg: TrainConfig = DEFAULT_TRAIN,
  lossFn: LossFn = "ce",
): number {
  const g = zeroParams(p.cfg);
  let loss = 0;
  for (const ex of batch) {
    const cache = forward(p, ex.input);
    loss += lossValue(lossFn, cache.probs, ex.target) / batch.length;
    backward(cache, p, ex.target, g, lossFn);
  }
  scaleGrads(g, 1 / batch.length);
  if (cfg.clip > 0) clipGrad(g, cfg.clip);
  adam.step += 1;
  adamUpdate(p, g, adam.m, adam.v, cfg.adam, adam.step);
  return loss;
}

/** Average loss over the whole dataset (for a smooth loss curve). */
export function evalLoss(p: Params, dataset: Example[], lossFn: LossFn = "ce"): number {
  let sum = 0;
  for (const ex of dataset) {
    const cache = forward(p, ex.input);
    sum += lossValue(lossFn, cache.probs, ex.target);
  }
  return sum / dataset.length;
}
