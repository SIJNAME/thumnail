import axios from "axios";
import { env } from "../config/env";
import { withRetry } from "../utils/retry";
import { ExternalServiceError, TimeoutError } from "../utils/errors";

export async function submitWorkflow(prompt: unknown): Promise<string> {
  return withRetry(async () => {
    try {
      const response = await axios.post(`${env.COMFY_URL}/prompt`, { prompt });
      return response.data.prompt_id as string;
    } catch {
      throw new ExternalServiceError("Comfy submit failed", "comfy");
    }
  });
}

export async function waitForComfyImage(promptId: string, maxAttempts = 45, delayMs = 2000): Promise<string> {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    await new Promise((r) => setTimeout(r, delayMs));
    try {
      const history = await axios.get(`${env.COMFY_URL}/history/${promptId}`);
      const outputs = history.data[promptId]?.outputs;
      if (!outputs) continue;
      for (const nodeId of Object.keys(outputs)) {
        if (outputs[nodeId].images?.length) return outputs[nodeId].images[0].filename;
      }
    } catch {
      // continue polling
    }
  }
  throw new TimeoutError("Comfy image generation timeout");
}

export async function comfyHealthcheck(): Promise<boolean> {
  try {
    await axios.get(`${env.COMFY_URL}/system_stats`, { timeout: 2000 });
    return true;
  } catch {
    return false;
  }
}
