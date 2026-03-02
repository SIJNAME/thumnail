import { Worker } from "bullmq";
import axios from "axios";
import { redis } from "../config/redis";
import { submitWorkflow, waitForComfyImage } from "../services/comfy.service";
import { getCache, setJobState } from "../services/cache.service";

const { matchHistogram, colorTransfer } = require("../../coloranalysis.js");

export const generateWorker = new Worker("generateThumbnailQueue", async (job) => {
  const { workflow, channelId, colorAdjust } = job.data as { workflow: Record<string, unknown>; channelId: string; colorAdjust?: boolean };
  await setJobState(job.id as string, { status: "active", progress: 15 });

  const promptId = await submitWorkflow(workflow);
  const fileName = await waitForComfyImage(promptId);
  const imageUrl = `${process.env.COMFY_URL}/view?filename=${fileName}`;
  const imageResp = await axios.get(imageUrl, { responseType: "arraybuffer" });
  let processed = Buffer.from(imageResp.data);

  const dna = await getCache<any>(`dna:${channelId}`);
  if (colorAdjust && dna?.avg_color_histogram_256) processed = await matchHistogram(processed, dna.avg_color_histogram_256);
  else if (colorAdjust && dna?.avg_color_stats) processed = await colorTransfer(processed, dna.avg_color_stats, 0.4);

  const result = { imageUrl, channelId, colorAdjust: Boolean(colorAdjust), bytes: processed.length };
  await setJobState(job.id as string, { status: "completed", result });
  return result;
}, { connection: redis });
