import OpenAI from "openai";
import pLimit from "p-limit";
import { env } from "../config/env";
import { logger } from "../config/logger";
import { withRetry } from "../utils/retry";
import { ExternalServiceError } from "../utils/errors";

const openai = new OpenAI({ apiKey: env.OPENAI_API_KEY });
const llmLimit = pLimit(5);
const visionLimit = pLimit(5);

export async function callLLM(messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[], json = false) {
  return llmLimit(() => withRetry(async () => {
    try {
      return await openai.chat.completions.create({
        model: env.OPENAI_MODEL,
        messages,
        ...(json ? { response_format: { type: "json_object" } as const } : {}),
      });
    } catch (err) {
      logger.error({ err }, "openai llm call failed");
      throw new ExternalServiceError("OpenAI LLM call failed", "openai");
    }
  }));
}

export async function callVision(messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[]) {
  return visionLimit(() => withRetry(async () => {
    try {
      return await openai.chat.completions.create({
        model: env.OPENAI_MODEL,
        messages,
        response_format: { type: "json_object" },
      });
    } catch (err) {
      logger.error({ err }, "openai vision call failed");
      throw new ExternalServiceError("OpenAI Vision call failed", "openai");
    }
  }));
}

export async function openaiHealthcheck(): Promise<boolean> {
  try {
    await openai.models.list();
    return true;
  } catch {
    return false;
  }
}
