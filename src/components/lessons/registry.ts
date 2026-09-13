import type { ComponentType } from "react";
import Lesson01 from "./01-tokens";
import Lesson02 from "./02-vocab";
import Lesson03 from "./03-embeddings";
import Lesson04 from "./04-position";
import Lesson05 from "./05-task";
import Lesson06 from "./06-attention";
import Lesson07 from "./07-multihead";
import Lesson08 from "./08-block";
import Lesson09 from "./09-softmax";
import Lesson10 from "./10-loss";
import Lesson11 from "./11-gd";
import Lesson12 from "./12-backprop";
import Lesson13 from "./13-optimizer";
import Lesson14 from "./14-loop";
import Lesson15 from "./15-generate";
import Lesson16 from "./16-recap";

export const LESSON_COMPONENTS: Record<string, ComponentType> = {
  "text-into-tokens": Lesson01,
  "vocabulary-token-ids": Lesson02,
  embeddings: Lesson03,
  "positional-embeddings": Lesson04,
  "predict-next-token": Lesson05,
  "self-attention": Lesson06,
  "multi-head-attention": Lesson07,
  "transformer-block": Lesson08,
  "logits-softmax": Lesson09,
  loss: Lesson10,
  "gradient-descent": Lesson11,
  backpropagation: Lesson12,
  optimizer: Lesson13,
  "training-loop": Lesson14,
  "generate-text": Lesson15,
  recap: Lesson16,
};
