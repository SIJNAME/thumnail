import { Request, Response } from "express";
import { analyzeChannelQueue, generateThumbnailQueue } from "../queue";
import { z } from "zod";
import { getJobState } from "../services/cache.service";

const analyzeSchema = z.object({
  channelId: z.string().min(1),
  thumbnails: z.array(z.string().url()).default([]),
});

const generateSchema = z.object({
  workflow: z.record(z.any()),
  channelId: z.string().min(1),
  colorAdjust: z.boolean().optional(),
});

export async function enqueueAnalyzeChannel(req: Request, res: Response) {
  const parsed = analyzeSchema.parse(req.body);
  const job = await analyzeChannelQueue.add("analyze-channel", parsed, { attempts: 3, backoff: { type: "exponential", delay: 1000 } });
  res.json({ jobId: job.id, queue: "analyzeChannelQueue" });
}

export async function enqueueGenerateThumbnail(req: Request, res: Response) {
  const parsed = generateSchema.parse(req.body);
  const job = await generateThumbnailQueue.add("generate-thumbnail", parsed, { attempts: 3, backoff: { type: "exponential", delay: 1000 } });
  res.json({ jobId: job.id, queue: "generateThumbnailQueue" });
}

export async function getJobStatus(req: Request, res: Response) {
  const state = await getJobState(req.params.jobId);
  res.json(state || { status: "waiting" });
}
