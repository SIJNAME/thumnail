import { Router } from "express";
import { enqueueAnalyzeChannel, enqueueGenerateThumbnail, getJobStatus } from "../controllers/jobs.controller";
import { health, metrics } from "../controllers/health.controller";

export const router = Router();

router.get("/health", health);
router.get("/metrics", metrics);
router.post("/analyze-channel", enqueueAnalyzeChannel);
router.post("/generate-thumbnail", enqueueGenerateThumbnail);
router.get("/jobs/:jobId", getJobStatus);
