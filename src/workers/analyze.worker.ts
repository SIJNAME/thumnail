import { Worker } from "bullmq";
import axios from "axios";
import { redis } from "../config/redis";
import { setCache, setJobState } from "../services/cache.service";
import { callVision } from "../services/openai.service";
import { sanitizeUrl } from "../utils/url";

const { getHistogram } = require("../../coloranalysis.js");

export const analyzeWorker = new Worker("analyzeChannelQueue", async (job) => {
  const { channelId, thumbnails } = job.data as { channelId: string; thumbnails: string[] };
  await setJobState(job.id as string, { status: "active", progress: 10 });

  const samples = [] as unknown[];
  for (const t of thumbnails || []) {
    const url = sanitizeUrl(t);
    const image = await axios.get(url, { responseType: "arraybuffer" });
    const buffer = Buffer.from(image.data);
    const hist = await getHistogram(buffer);
    const vision = await callVision([
      { role: "system", content: "Analyze thumbnail and return JSON only." },
      { role: "user", content: [{ type: "text", text: "Return visual_ctr_score, face_bbox, text_bbox" }, { type: "image_url", image_url: { url: `data:image/png;base64,${buffer.toString("base64")}` } }] },
    ]);
    samples.push({ hist, vision: vision.choices?.[0]?.message?.content });
  }

  const result = { channelId, sample_size: samples.length, analyzed_at: new Date().toISOString() };
  await setCache(`dna:${channelId}`, result, 24 * 60 * 60);
  await setJobState(job.id as string, { status: "completed", result });
  return result;
}, { connection: redis });
