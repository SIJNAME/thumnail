import "../config/env";
import { logger } from "../config/logger";
import { analyzeWorker } from "./analyze.worker";
import { generateWorker } from "./generate.worker";

logger.info("workers started");

process.on("SIGTERM", async () => {
  await Promise.all([analyzeWorker.close(), generateWorker.close()]);
  process.exit(0);
});
