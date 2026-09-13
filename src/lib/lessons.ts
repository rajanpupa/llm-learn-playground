export interface Lesson {
  slug: string;
  number: number; // 1-based
  title: string;
  summary: string;
  category: string;
}

export const LESSONS: Lesson[] = [
  {
    slug: "text-into-tokens",
    number: 1,
    title: "Text into Tokens",
    summary: "A model can't read letters — it reads numbers. We split text into tokens.",
    category: "Foundations",
  },
  {
    slug: "vocabulary-token-ids",
    number: 2,
    title: "The Vocabulary & Token IDs",
    summary: "Every token gets an integer id, then a one-hot vector.",
    category: "Foundations",
  },
  {
    slug: "embeddings",
    number: 3,
    title: "Embeddings: Words as Vectors",
    summary: "Token ids become dense vectors that capture meaning.",
    category: "Foundations",
  },
  {
    slug: "positional-embeddings",
    number: 4,
    title: "Position: Order Matters",
    summary: "The same word in different places needs to look different.",
    category: "Foundations",
  },
  {
    slug: "predict-next-token",
    number: 5,
    title: "The Core Task: Predict the Next Token",
    summary: "Every LLM is trained on one job — guess what comes next.",
    category: "Foundations",
  },
  {
    slug: "self-attention",
    number: 6,
    title: "Self-Attention: Tokens Look at Each Other",
    summary: "Query, Key, Value — how a token decides what to focus on.",
    category: "The Architecture",
  },
  {
    slug: "multi-head-attention",
    number: 7,
    title: "Multi-Head Attention",
    summary: "Several attention heads in parallel, each noticing something different.",
    category: "The Architecture",
  },
  {
    slug: "transformer-block",
    number: 8,
    title: "The Transformer Block",
    summary: "Attention + a feed-forward network, stacked with residuals.",
    category: "The Architecture",
  },
  {
    slug: "logits-softmax",
    number: 9,
    title: "Logits & Softmax: Scores → Probabilities",
    summary: "The model's raw scores are turned into probabilities.",
    category: "The Architecture",
  },
  {
    slug: "loss",
    number: 10,
    title: "Loss: Measuring Mistakes",
    summary: "Cross-entropy compares the model's guess to the true answer.",
    category: "Learning",
  },
  {
    slug: "gradient-descent",
    number: 11,
    title: "Gradient Descent: Walking Downhill",
    summary: "Training = nudging weights to reduce the loss, one step at a time.",
    category: "Learning",
  },
  {
    slug: "backpropagation",
    number: 12,
    title: "Backpropagation: Who's Responsible?",
    summary: "The chain rule traces the loss back to every weight.",
    category: "Learning",
  },
  {
    slug: "optimizer",
    number: 13,
    title: "The Optimizer: Adam, LR & Clipping",
    summary: "Smart updates with momentum, a learning rate, and safety rails.",
    category: "Learning",
  },
  {
    slug: "training-loop",
    number: 14,
    title: "The Training Loop: Putting It Together",
    summary: "Feed, measure, update, repeat — watch ToyGPT learn live.",
    category: "Learning",
  },
  {
    slug: "generate-text",
    number: 15,
    title: "Generating Text: Sampling & Temperature",
    summary: "The trained model writes new text, one token at a time.",
    category: "Inference & Wrap-up",
  },
  {
    slug: "recap",
    number: 16,
    title: "Recap: The Whole Pipeline",
    summary: "Every stage, from raw text to a talking model, in one view.",
    category: "Inference & Wrap-up",
  },
];

export function getLesson(slug: string): Lesson | undefined {
  return LESSONS.find((l) => l.slug === slug);
}

export function getNextLesson(slug: string): Lesson | undefined {
  const i = LESSONS.findIndex((l) => l.slug === slug);
  return i >= 0 && i < LESSONS.length - 1 ? LESSONS[i + 1] : undefined;
}

export function getPrevLesson(slug: string): Lesson | undefined {
  const i = LESSONS.findIndex((l) => l.slug === slug);
  return i > 0 ? LESSONS[i - 1] : undefined;
}

export function formatLessonNumber(n: number): string {
  return String(n).padStart(2, "0");
}
