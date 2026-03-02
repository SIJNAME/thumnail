import express from "express";
import helmet from "helmet";
import cors from "cors";
import compression from "compression";
import rateLimit from "express-rate-limit";
import pinoHttp from "pino-http";
import { router } from "./routes";
import { logger } from "./config/logger";
import { requestIdMiddleware } from "./middleware/requestId";
import { responseTimeMiddleware } from "./middleware/responseTime";

export function buildApp() {
  const app = express();

  app.use(helmet());
  app.use(cors({ origin: true, credentials: false }));
  app.use(compression());
  app.use(express.json({ limit: "5mb" }));
  app.use(rateLimit({ windowMs: 60_000, max: 120 }));
  app.use(requestIdMiddleware);
  app.use(pinoHttp({ logger }));
  app.use(responseTimeMiddleware);

  app.use(router);

  app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
    logger.error({ err }, "unhandled error");
    res.status(500).json({ error: err.message || "Internal Server Error" });
  });

  return app;
}
