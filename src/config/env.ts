import "dotenv/config";
import { z } from "zod";

const envSchema = z.object({
  OPENAI_API_KEY: z.string().min(1),
  REDIS_URL: z.string().url(),
  COMFY_URL: z.string().url(),
  YOUTUBE_API_KEY: z.string().min(1),
  PORT: z.coerce.number().default(9000),
  OPENAI_MODEL: z.string().default("gpt-4o-mini"),
  DEBUG_PROMPT: z.string().optional(),
});

const parsed = envSchema.safeParse(process.env);
if (!parsed.success) {
  // Fail fast in production startup
  throw new Error(`Invalid environment configuration: ${parsed.error.message}`);
}

export const env = parsed.data;
