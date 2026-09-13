// The single running example threaded through every lesson.
// It is small enough that every number can be shown on screen.

export const CORPUS_LINES = [
  "the cat sat",
  "the dog sat",
  "the cat ran",
  "the dog ran",
] as const;

export const CORPUS = CORPUS_LINES.join("\n");

// --- Vocabulary -------------------------------------------------------------

export interface Token {
  id: number;
  char: string;
  display: string;
}

// Unique characters, in order of first appearance across the corpus.
export const VOCAB: Token[] = [
  { id: 0, char: "t", display: "t" },
  { id: 1, char: "h", display: "h" },
  { id: 2, char: "e", display: "e" },
  { id: 3, char: " ", display: "␣" }, // space
  { id: 4, char: "c", display: "c" },
  { id: 5, char: "a", display: "a" },
  { id: 6, char: "s", display: "s" },
  { id: 7, char: "\n", display: "⏎" }, // newline
  { id: 8, char: "d", display: "d" },
  { id: 9, char: "o", display: "o" },
  { id: 10, char: "g", display: "g" },
  { id: 11, char: "r", display: "r" },
  { id: 12, char: "n", display: "n" },
];

export const VOCAB_SIZE = VOCAB.length;

const CHAR_TO_ID: Record<string, number> = {};
for (const t of VOCAB) CHAR_TO_ID[t.char] = t.id;

export function charToId(char: string): number {
  const id = CHAR_TO_ID[char];
  if (id === undefined) throw new Error(`Unknown character: ${JSON.stringify(char)}`);
  return id;
}

/** Tokenize text into token ids. */
export function tokenize(text: string): number[] {
  return Array.from(text).map(charToId);
}

/** One-hot vector (length VOCAB_SIZE) with a 1 at `id`. */
export function oneHot(id: number): number[] {
  const v = new Array<number>(VOCAB_SIZE).fill(0);
  v[id] = 1;
  return v;
}

/** Pretty label for a token (uses ␣ for space, ⏎ for newline). */
export function tokenLabel(char: string): string {
  if (char === " ") return "␣";
  if (char === "\n") return "⏎";
  return char;
}

export const tokenDisplay = (id: number): string => VOCAB[id]?.display ?? "?";

/** ids → human-readable string (space / newline shown as glyphs). */
export function idsToLabel(ids: number[]): string {
  return ids.map((id) => tokenDisplay(id)).join("");
}

// --- Model constants (ToyGPT) ----------------------------------------------

export const EMBED_DIM = 8; // embedding size (tiny so we can print it)
export const BLOCK_SIZE = 8; // context window (tokens the model sees at once)
export const N_HEAD = 2; // attention heads
export const HEAD_DIM = EMBED_DIM / N_HEAD; // = 4
export const MLP_DIM = EMBED_DIM * 4; // = 32

// --- The dataset: sliding windows of input → target ------------------------

export interface Example {
  input: number[]; // BLOCK_SIZE token ids
  target: number[]; // BLOCK_SIZE token ids (input shifted left by one)
}

/** Build every (input, target) window from the corpus. */
export function buildDataset(): Example[] {
  const all = tokenize(CORPUS);
  const examples: Example[] = [];
  for (let i = 0; i + BLOCK_SIZE < all.length; i++) {
    examples.push({
      input: all.slice(i, i + BLOCK_SIZE),
      target: all.slice(i + 1, i + 1 + BLOCK_SIZE),
    });
  }
  return examples;
}

export const DATASET = buildDataset();

// The first line, pre-tokenized, used repeatedly in examples.
export const FIRST_LINE = "the cat sat";
export const FIRST_LINE_IDS = tokenize(FIRST_LINE); // 11 ids

// --- Configurable model -----------------------------------------------------

export interface ModelConfig {
  vocabSize: number;
  embedDim: number; // "the size of the vector" in the playground
  blockSize: number; // context window
  nHead: number; // attention heads
}

/** The fixed tiny model used across the lessons. */
export const TOY_CONFIG: ModelConfig = {
  vocabSize: VOCAB_SIZE,
  embedDim: EMBED_DIM,
  blockSize: BLOCK_SIZE,
  nHead: N_HEAD,
};

// --- Generic vocabulary / dataset from arbitrary text ----------------------

/** Unique characters in `text`, in order of first appearance. */
export function buildVocab(text: string): Token[] {
  const seen = new Set<string>();
  const out: Token[] = [];
  for (const ch of Array.from(text)) {
    if (seen.has(ch)) continue;
    seen.add(ch);
    out.push({ id: out.length, char: ch, display: tokenLabel(ch) });
  }
  return out;
}

/** Tokenize `text` against a specific vocabulary. */
export function tokenizeWith(vocab: Token[], text: string): number[] {
  const map = new Map<string, number>();
  for (const t of vocab) map.set(t.char, t.id);
  return Array.from(text).map((ch) => {
    const id = map.get(ch);
    if (id === undefined) throw new Error(`Unknown character: ${JSON.stringify(ch)}`);
    return id;
  });
}

/** Build sliding-window (input, target) examples from arbitrary text. */
export function buildDatasetFromText(text: string, blockSize: number): Example[] {
  const vocab = buildVocab(text);
  const all = tokenizeWith(vocab, text);
  const examples: Example[] = [];
  for (let i = 0; i + blockSize < all.length; i++) {
    examples.push({
      input: all.slice(i, i + blockSize),
      target: all.slice(i + 1, i + 1 + blockSize),
    });
  }
  return examples;
}
