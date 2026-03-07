import { Request, Response } from "express";
import { redis } from "../config/redis";
import { openaiHealthcheck } from "../services/openai.service";
import { comfyHealthcheck } from "../services/comfy.service";
import { register } from "prom-client";

export async function health(req: Request, res: Response) {
  const redisOk = redis.status === "ready";
  const [openaiOk, comfyOk] = await Promise.all([openaiHealthcheck(), comfyHealthcheck()]);

  res.status(redisOk && openaiOk && comfyOk ? 200 : 503).json({
    redis: redisOk,
    openai: openaiOk,
    comfy: comfyOk,
  });
}

export async function metrics(_req: Request, res: Response) {
  res.setHeader("Content-Type", register.contentType);
  res.send(await register.metrics());
}
