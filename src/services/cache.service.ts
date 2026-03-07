import { redis } from "../config/redis";

export async function setCache<T>(key: string, value: T, ttlSec: number): Promise<void> {
  await redis.set(key, JSON.stringify(value), "EX", ttlSec);
}

export async function getCache<T>(key: string): Promise<T | null> {
  const raw = await redis.get(key);
  return raw ? (JSON.parse(raw) as T) : null;
}

export async function setJobState(jobId: string, state: unknown): Promise<void> {
  await setCache(`job:${jobId}`, state, 24 * 60 * 60);
}

export async function getJobState(jobId: string): Promise<unknown> {
  return getCache(`job:${jobId}`);
}
