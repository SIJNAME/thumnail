import { Request, Response, NextFunction } from "express";
import { logger } from "../config/logger";

export function responseTimeMiddleware(req: Request, res: Response, next: NextFunction): void {
  const start = process.hrtime.bigint();
  res.on("finish", () => {
    const durationMs = Number(process.hrtime.bigint() - start) / 1_000_000;
    logger.info({
      requestId: req.headers["x-request-id"],
      method: req.method,
      path: req.originalUrl,
      status: res.statusCode,
      durationMs: Number(durationMs.toFixed(2)),
    }, "request completed");
  });
  next();
}
