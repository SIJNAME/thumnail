import { Queue, QueueEvents } from "bullmq";
import { redis } from "../config/redis";

export const analyzeChannelQueue = new Queue("analyzeChannelQueue", { connection: redis });
export const generateThumbnailQueue = new Queue("generateThumbnailQueue", { connection: redis });

export const analyzeEvents = new QueueEvents("analyzeChannelQueue", { connection: redis });
export const generateEvents = new QueueEvents("generateThumbnailQueue", { connection: redis });
