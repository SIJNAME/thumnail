import { buildApp } from "./app";
import { env } from "./config/env";
import { logger } from "./config/logger";
import { redis } from "./config/redis";

const app = buildApp();
const server = app.listen(env.PORT, () => {
  logger.info({ port: env.PORT }, "API server started");
});

async function shutdown(signal: string) {
  logger.info({ signal }, "graceful shutdown started");
  server.close(async () => {
    await redis.quit();
    logger.info("graceful shutdown completed");
    process.exit(0);
  });
}

process.on("SIGINT", () => void shutdown("SIGINT"));
process.on("SIGTERM", () => void shutdown("SIGTERM"));
