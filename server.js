require("dotenv").config();

if (!process.env.OPENAI_API_KEY) {
  console.error("Missing OPENAI_API_KEY");
  process.exit(1);
}

if (!process.env.YOUTUBE_API_KEY) {
  console.error("Missing YOUTUBE_API_KEY");
  process.exit(1);
}

const sharp = require("sharp");
const express = require("express");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const OpenAI = require("openai");
const { matchHistogram, getHistogram } = require("./coloranalysis");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const app = express();
app.use(express.json({ limit: "20mb" }));

const COMFY_URL = process.env.COMFY_URL || "http://127.0.0.1:8188";
const YOUTUBE_API_KEY = process.env.YOUTUBE_API_KEY;
const DNA_DIR = path.join(__dirname, "channel_dna");
const TEMP_DIR = path.join(__dirname, "temp");
const DEBUG = process.env.DEBUG_PROMPT === "1";
const GRID_W = 64;
const GRID_H = 36;

if (!fs.existsSync(DNA_DIR)) fs.mkdirSync(DNA_DIR);
if (!fs.existsSync(TEMP_DIR)) fs.mkdirSync(TEMP_DIR);

const analyzeCache = new Map();
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

class Semaphore {
  constructor(max) {
    this.max = max;
    this.current = 0;
    this.waiters = [];
  }
  async acquire() {
    if (this.current < this.max) {
      this.current += 1;
      return;
    }
    await new Promise((resolve) => this.waiters.push(resolve));
    this.current += 1;
  }
  release() {
    this.current = Math.max(0, this.current - 1);
    const next = this.waiters.shift();
    if (next) next();
  }
}

const comfySemaphore = new Semaphore(2);

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function safeJsonParse(raw, fallback = null) {
  try {
    return JSON.parse(raw);
  } catch (_err) {
    return fallback;
  }
}

function quotaAwareError(err) {
  const status = err?.response?.status;
  const reason = err?.response?.data?.error?.errors?.[0]?.reason;
  if (status === 403 && ["quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"].includes(reason)) {
    return `YouTube quota exceeded (${reason})`;
  }
  return null;
}

async function youtubeGet(url, params) {
  try {
    return await axios.get(url, { params });
  } catch (err) {
    const msg = quotaAwareError(err);
    if (msg) throw new Error(msg);
    throw err;
  }
}

async function getMeanStd(imageBuffer) {
  const { data, info } = await sharp(imageBuffer).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const channels = info.channels;
  const pixels = info.width * info.height;
  const stats = {
    r: { sum: 0, sqSum: 0 },
    g: { sum: 0, sqSum: 0 },
    b: { sum: 0, sqSum: 0 },
  };

  for (let i = 0; i < data.length; i += channels) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    stats.r.sum += r;
    stats.g.sum += g;
    stats.b.sum += b;
    stats.r.sqSum += r * r;
    stats.g.sqSum += g * g;
    stats.b.sqSum += b * b;
  }

  const calc = (c) => {
    const mean = c.sum / pixels;
    const variance = c.sqSum / pixels - mean * mean;
    return { mean, std: Math.sqrt(Math.max(variance, 1)) };
  };

  return { r: calc(stats.r), g: calc(stats.g), b: calc(stats.b) };
}

async function colorTransfer(imageBuffer, targetStats, weight = 0.4) {
  const source = await getMeanStd(imageBuffer);
  const safe = (v) => (v === 0 ? 1 : v);
  const scaleR = 1 + weight * (targetStats.r.std / safe(source.r.std) - 1);
  const scaleG = 1 + weight * (targetStats.g.std / safe(source.g.std) - 1);
  const scaleB = 1 + weight * (targetStats.b.std / safe(source.b.std) - 1);
  const shiftR = weight * (targetStats.r.mean - source.r.mean);
  const shiftG = weight * (targetStats.g.mean - source.g.mean);
  const shiftB = weight * (targetStats.b.mean - source.b.mean);
  return sharp(imageBuffer).linear([scaleR, scaleG, scaleB], [shiftR, shiftG, shiftB]).toBuffer();
}

function mostCommon(arr) {
  if (!arr?.length) return null;
  const counts = {};
  arr.forEach((item) => {
    if (!item) return;
    counts[item] = (counts[item] || 0) + 1;
  });
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || null;
}

function loadChannelDNA(channelId) {
  const dnaPath = path.join(DNA_DIR, `${channelId}.json`);
  if (!fs.existsSync(dnaPath)) return null;
  return safeJsonParse(fs.readFileSync(dnaPath, "utf-8"), null);
}

function ensureMatrix(map, h = GRID_H, w = GRID_W) {
  if (!Array.isArray(map) || map.length !== h) return null;
  for (const row of map) {
    if (!Array.isArray(row) || row.length !== w) return null;
    for (const v of row) if (typeof v !== "number") return null;
  }
  return map;
}

function validateDnaSchema(dna) {
  if (!dna || typeof dna !== "object") return false;
  const requiredNums = ["sample_size", "avg_visual_ctr", "subject_position_x", "emotion_intensity_avg", "contrast_strength"];
  if (!requiredNums.every((k) => typeof dna[k] === "number")) return false;
  if (!dna.avg_color_histogram_256?.rHist || dna.avg_color_histogram_256.rHist.length !== 256) return false;
  if (!ensureMatrix(dna.subject_density_map) || !ensureMatrix(dna.negative_space_map)) return false;
  return true;
}

function saveChannelDNA(channelId, dna) {
  if (!validateDnaSchema(dna)) throw new Error("DNA schema validation failed");
  const dnaPath = path.join(DNA_DIR, `${channelId}.json`);
  fs.writeFileSync(dnaPath, JSON.stringify(dna, null, 2));
}

function toGridIndex(xNorm, yNorm, wNorm, hNorm) {
  const x1 = clamp(Math.floor(xNorm * GRID_W), 0, GRID_W - 1);
  const y1 = clamp(Math.floor(yNorm * GRID_H), 0, GRID_H - 1);
  const x2 = clamp(Math.ceil((xNorm + wNorm) * GRID_W), x1 + 1, GRID_W);
  const y2 = clamp(Math.ceil((yNorm + hNorm) * GRID_H), y1 + 1, GRID_H);
  return { x1, y1, x2, y2 };
}

function createGrid(fill = 0) {
  return Array.from({ length: GRID_H }, () => Array(GRID_W).fill(fill));
}

function addBoxToGrid(grid, bbox, value = 1) {
  if (!bbox) return;
  const { x1, y1, x2, y2 } = toGridIndex(bbox.x || 0, bbox.y || 0, bbox.w || 0, bbox.h || 0);
  for (let y = y1; y < y2; y += 1) {
    for (let x = x1; x < x2; x += 1) grid[y][x] += value;
  }
}

function normalizeGrid(grid, divisor) {
  return grid.map((row) => row.map((v) => Number((v / Math.max(1, divisor)).toFixed(4))));
}

function highNegativeSpaceZones(negativeSpaceMap, threshold = 0.6) {
  const zones = [];
  for (let y = 0; y < GRID_H; y += 1) {
    for (let x = 0; x < GRID_W; x += 1) {
      if ((negativeSpaceMap[y]?.[x] || 0) > threshold) zones.push({ x, y });
    }
  }
  if (!zones.length) return "center";
  const avgX = zones.reduce((s, z) => s + z.x, 0) / zones.length;
  if (avgX < GRID_W / 3) return "left";
  if (avgX > (2 * GRID_W) / 3) return "right";
  return "center";
}

function bboxOverlapRatio(a, b) {
  if (!a || !b) return 0;
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const ix1 = Math.max(a.x, b.x);
  const iy1 = Math.max(a.y, b.y);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const textArea = Math.max(1e-6, b.w * b.h);
  return inter / textArea;
}

async function createControlNetMaskFromDensityMap(subjectDensityMap, outPath) {
  const raw = Buffer.alloc(GRID_W * GRID_H);
  for (let y = 0; y < GRID_H; y += 1) {
    for (let x = 0; x < GRID_W; x += 1) {
      const v = subjectDensityMap[y]?.[x] || 0;
      raw[y * GRID_W + x] = v > 0.5 ? 255 : 0;
    }
  }

  await sharp(raw, { raw: { width: GRID_W, height: GRID_H, channels: 1 } })
    .resize(1024, 576, { kernel: "nearest" })
    .png()
    .toFile(outPath);
  return outPath;
}

function normalizeBBox(b) {
  const fix = (v) => {
    if (typeof v !== "number" || Number.isNaN(v)) return 0;
    let x = v > 1.5 ? v / 100 : v;
    x = clamp(x, 0, 1);
    return Number(x.toFixed(3));
  };
  return { x: fix(b?.x), y: fix(b?.y), w: fix(b?.w), h: fix(b?.h) };
}

function inferTypographyStats(vision) {
  const txt = String(vision?.ocr_text || "");
  const letters = txt.match(/[A-Za-z]/g) || [];
  const uppers = txt.match(/[A-Z]/g) || [];
  const words = txt.match(/[A-Za-z0-9]+/g) || [];
  return {
    uppercase_ratio: letters.length ? uppers.length / letters.length : 0,
    dominant_text_outline: (vision?.readability_score || 0) > 70 ? 1 : 0,
    dominant_font_weight: (vision?.readability_score || 0) > 75 ? "bold" : (vision?.readability_score || 0) > 50 ? "medium" : "light",
    text_area_ratio: Math.max(0, (vision?.text_bbox?.w || 0) * (vision?.text_bbox?.h || 0)),
    avg_word_count: words.length,
  };
}

async function getChannelVideos(channelId) {
  const response = await youtubeGet("https://www.googleapis.com/youtube/v3/search", {
    part: "snippet",
    channelId,
    maxResults: 20,
    order: "date",
    type: "video",
    key: YOUTUBE_API_KEY,
  });

  return response.data.items.map((item) => ({
    videoId: item.id.videoId,
    thumbnail: item.snippet.thumbnails.high?.url || item.snippet.thumbnails.default?.url,
  }));
}

async function getChannelVideoDetails(channelId) {
  const search = await youtubeGet("https://www.googleapis.com/youtube/v3/search", {
    part: "snippet",
    channelId,
    maxResults: 15,
    order: "date",
    type: "video",
    key: YOUTUBE_API_KEY,
  });

  const videoIds = search.data.items.map((v) => v.id.videoId).join(",");
  const videos = await youtubeGet("https://www.googleapis.com/youtube/v3/videos", {
    part: "snippet,statistics",
    id: videoIds,
    key: YOUTUBE_API_KEY,
  });

  return videos.data.items.map((v) => ({
    title: v.snippet.title,
    description: v.snippet.description,
    tags: v.snippet.tags || [],
    categoryId: v.snippet.categoryId,
    viewCount: v.statistics.viewCount,
    likeCount: v.statistics.likeCount,
  }));
}

async function analyzeChannelNiche(videoList) {
  const combinedText = videoList
    .map((v) => `Title: ${v.title}\nDescription: ${v.description}\nTags: ${v.tags.join(", ")}`)
    .join("\n\n");

  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: "You are a YouTube channel strategist. Analyze channel niche deeply." },
      {
        role: "user",
        content: `Analyze this channel content and return JSON:\n{\n"niche":"","primary_topic":"","common_objects":[],"emotion_baseline":"","style_profile":"","visual_theme":""\n}\n\n${combinedText}`,
      },
    ],
  });

  return safeJsonParse(response.choices[0].message.content, {
    niche: "general",
    primary_topic: "general",
    common_objects: [],
    emotion_baseline: "neutral",
    style_profile: "general",
    visual_theme: "general",
  });
}

async function analyzeImageWithVision(imageBuffer) {
  const base64Image = imageBuffer.toString("base64");
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content:
          "You are a YouTube thumbnail CTR expert. Return JSON only. All ratio and bbox coordinates in 0..1.",
      },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Return fields: visual_ctr_score, face_score, emotion_score, color_score, readability_score, face_ratio, subject_position_x, negative_space_ratio, camera_distance_estimate, horizon_line_estimate, color_saturation_level, contrast_strength, emotion_intensity_avg, background_blur_level, object_density, lighting_style, composition_style, text_position, realism_level, art_style, render_type, subject_distortion_type, color_palette_type, face_bbox, text_bbox, color_stats, avg_rgb, brightness_mean, saturation_mean, ocr_text",
          },
          { type: "image_url", image_url: { url: `data:image/png;base64,${base64Image}` } },
        ],
      },
    ],
  });

  const parsed = safeJsonParse(response.choices[0].message.content, null);
  if (!parsed) return null;

  parsed.face_bbox = normalizeBBox(parsed.face_bbox);
  parsed.text_bbox = normalizeBBox(parsed.text_bbox);
  parsed.face_ratio = clamp(Number(parsed.face_ratio || 0), 0, 1);
  parsed.subject_position_x = clamp(Number(parsed.subject_position_x || 0), 0, 1);
  parsed.negative_space_ratio = clamp(Number(parsed.negative_space_ratio || 0), 0, 1);
  parsed.camera_distance_estimate = clamp(Number(parsed.camera_distance_estimate || 0), 0, 1);
  parsed.horizon_line_estimate = clamp(Number(parsed.horizon_line_estimate || 0), 0, 1);
  return parsed;
}

async function analyzeContent(data, channelDNA) {
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `You are an elite YouTube CTR strategist. Channel baseline: ${channelDNA?.niche_profile?.niche || "general"}`,
      },
      {
        role: "user",
        content: `Analyze video and return JSON with category, template, game, specific_game_elements, visual_objects, environment_type, mood, focus, style, layout, text_hook, hook_strength, emotion_intensity, ctr_score, text_variations.\nTitle:${data.title}\nDescription:${data.description}`,
      },
    ],
  });
  return safeJsonParse(response.choices[0].message.content, null);
}

function mergeDNA(context, channelDNA) {
  const n = channelDNA?.niche_profile || {};
  return `Channel Style Baseline:
Lighting: ${channelDNA?.dominant_lighting_style || "cinematic"}
Composition: ${channelDNA?.dominant_composition_style || "centered"}
Art style: ${channelDNA?.dominant_art_style || "photorealistic"}
Color palette: ${channelDNA?.dominant_color_palette || "vivid"}
Render type: ${channelDNA?.dominant_render_type || "digital"}
Niche: ${n.niche || "general"}
Primary topic: ${n.primary_topic || "general"}
Realism: ${channelDNA?.avg_realism_level || 50}`;
}

function buildPrompt(context) {
  const dna = context.dna || {};
  const textPosition = dna.dominant_text_position || "right";
  const textB = dna.text_bbox_avg || { x: 0.7, y: 0.15, w: 0.25, h: 0.18 };
  const faceB = dna.face_bbox_avg || { x: 0.25, y: 0.2, w: 0.35, h: 0.55 };
  const zone = highNegativeSpaceZones(dna.negative_space_map || createGrid(0.5), 0.6);

  const typographyRules = [];
  if ((dna.uppercase_ratio || 0) > 0.7) typographyRules.push("Use ALL CAPS bold text");
  else typographyRules.push("Use mixed-case readable bold text");
  if (dna.dominant_text_outline) typographyRules.push("Text must have strong outline stroke");
  typographyRules.push(`Target text area ratio: ${(dna.text_area_ratio || 0.1).toFixed(2)}`);
  typographyRules.push(`Text position approx x=${textB.x.toFixed(2)}, y=${textB.y.toFixed(2)}`);
  typographyRules.push("Text must not overlap face region");

  const expressionRule = (dna.avg_emotion_score || 0) > 70
    ? "Highly expressive exaggerated facial emotion"
    : "Moderate natural facial expression";

  return `${mergeDNA(context, dna)}
-----------------------------------
Scene:
Main subject: ${context.focus}
Mood: ${context.mood}
Subject positioned around X=${(dna.subject_position_x || 0.5).toFixed(2)}
Camera distance similar to baseline (${(dna.camera_distance_estimate || 0.5).toFixed(2)})
Maintain composition style: ${dna.dominant_composition_style || "balanced"}
Place text in high negative space zones (${zone})
Avoid placing main subject in high density zones
${expressionRule}
-----------------------------------
Typography:
${typographyRules.join("\n")}
-----------------------------------
Thumbnail text: "${context.text_hook}"`;
}

function buildNegativePrompt() {
  return "blurry, low quality, bad anatomy, watermark, text artifacts, overlapping text on face";
}

function clip01(v) {
  return clamp(Number(v || 0), 0, 1);
}

function updateDNAFromWinner(channelId, winner, channelDNA) {
  if (!channelDNA || !winner) return;
  const alpha = 0.15;
  const dnaPath = path.join(DNA_DIR, `${channelId}.json`);

  const blend = (oldV, newV) => Number((oldV + alpha * (newV - oldV)).toFixed(4));

  channelDNA.sample_size = Math.min(200, (channelDNA.sample_size || 0) + 1);
  channelDNA.avg_visual_ctr = Math.round(blend(channelDNA.avg_visual_ctr || 50, winner.vision_score || 50));
  channelDNA.subject_position_x = blend(channelDNA.subject_position_x || 0.5, winner.subject_position_x || channelDNA.subject_position_x || 0.5);
  channelDNA.emotion_intensity_avg = blend(channelDNA.emotion_intensity_avg || 50, winner.emotion_score || channelDNA.emotion_intensity_avg || 50);
  channelDNA.color_saturation_level = blend(channelDNA.color_saturation_level || 50, winner.color_score || channelDNA.color_saturation_level || 50);
  channelDNA.contrast_strength = blend(channelDNA.contrast_strength || 50, winner.readability_score || channelDNA.contrast_strength || 50);
  channelDNA.negative_space_ratio = blend(channelDNA.negative_space_ratio || 0.5, winner.negative_space_ratio || channelDNA.negative_space_ratio || 0.5);

  fs.writeFileSync(dnaPath, JSON.stringify(channelDNA, null, 2));
}

async function getYouTubeData(videoId) {
  const response = await youtubeGet("https://www.googleapis.com/youtube/v3/videos", {
    part: "snippet,statistics",
    id: videoId,
    key: YOUTUBE_API_KEY,
  });
  if (!response.data.items.length) throw new Error("Video not found");
  const video = response.data.items[0];
  return {
    title: video.snippet.title,
    description: video.snippet.description,
    tags: video.snippet.tags || [],
    categoryId: video.snippet.categoryId,
    viewCount: video.statistics.viewCount,
    likeCount: video.statistics.likeCount,
  };
}

function calculateFinalScore(analysis) {
  return analysis.ctr_score * 0.5 + analysis.hook_strength * 10 * 0.3 + analysis.emotion_intensity * 10 * 0.2;
}

async function getChannelIdFromVideo(videoId) {
  const response = await youtubeGet("https://www.googleapis.com/youtube/v3/videos", {
    part: "snippet",
    id: videoId,
    key: YOUTUBE_API_KEY,
  });
  if (!response.data.items?.length) throw new Error("Video not found");
  return response.data.items[0].snippet.channelId;
}

function findNodeByType(workflow, type) {
  return Object.keys(workflow).find((key) => workflow[key].class_type === type);
}

function resolvePromptNodes(workflow) {
  const clipNodes = Object.keys(workflow).filter((k) => workflow[k].class_type === "CLIPTextEncode");
  if (clipNodes.length < 2) return { positiveNode: null, negativeNode: null };

  const looksNegative = (node) => {
    const title = String(node?._meta?.title || "").toLowerCase();
    const t = String(node?.inputs?.text || "").toLowerCase();
    return title.includes("negative")
      || t.includes("low quality")
      || t.includes("blurry")
      || t.includes("low contrast")
      || t.includes("bad anatomy");
  };

  const negativeNode = clipNodes.find((k) => looksNegative(workflow[k])) || clipNodes[1];
  const positiveNode = clipNodes.find((k) => k !== negativeNode) || clipNodes[0];
  return { positiveNode, negativeNode };
}

async function waitForComfyImage(promptId, maxAttempts = 45, delayMs = 2000) {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    await new Promise((r) => setTimeout(r, delayMs));
    try {
      const history = await axios.get(`${COMFY_URL}/history/${promptId}`);
      const outputs = history.data[promptId]?.outputs;
      if (!outputs) continue;
      for (const nodeId of Object.keys(outputs)) {
        if (outputs[nodeId].images?.length) return outputs[nodeId].images[0].filename;
      }
    } catch (_err) {
      // continue polling
    }
  }
  return null;
}

app.get("/analyze-channel", async (req, res) => {
  try {
    const channelId = req.query.channel_id;
    if (!channelId) return res.json({ error: "channel_id is required" });

    const cacheKey = `analyze:${channelId}`;
    const cached = analyzeCache.get(cacheKey);
    if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
      return res.json({ status: "dna_cached", dna: cached.data });
    }

    const videoDetails = await getChannelVideoDetails(channelId);
    const nicheProfile = await analyzeChannelNiche(videoDetails);
    const videos = await getChannelVideos(channelId);
    if (!videos.length) return res.json({ error: "No videos found" });

    const oldDNA = loadChannelDNA(channelId);

    const totals = {
      total: 0,
      face: 0,
      emotion: 0,
      color: 0,
      readability: 0,
      faceRatio: 0,
      subjectPos: 0,
      saturation: 0,
      contrast: 0,
      emotionIntensity: 0,
      blur: 0,
      objectDensity: 0,
      brightness: 0,
      saturationMean: 0,
      realism: 0,
      negativeSpace: 0,
      horizon: 0,
      cameraDistance: 0,
      uppercaseRatio: 0,
      textAreaRatio: 0,
      avgWordCount: 0,
    };

    const avgRGBTotal = [0, 0, 0];
    const sumR = new Array(256).fill(0);
    const sumG = new Array(256).fill(0);
    const sumB = new Array(256).fill(0);
    let histogramCount = 0;

    const lightingStyles = [];
    const compositionStyles = [];
    const textPositions = [];
    const artStyles = [];
    const renderTypes = [];
    const distortionTypes = [];
    const colorPalettes = [];
    const fontWeights = [];
    const textOutlines = [];

    const faceBBoxTotal = { x: 0, y: 0, w: 0, h: 0 };
    const textBBoxTotal = { x: 0, y: 0, w: 0, h: 0 };

    const occupancyAccum = createGrid(0);
    const analyzed = [];

    for (const video of videos) {
      try {
        const imageResp = await axios.get(video.thumbnail, { responseType: "arraybuffer" });
        let processedImage = Buffer.from(imageResp.data);

        const realHist = await getHistogram(processedImage);
        for (let i = 0; i < 256; i += 1) {
          sumR[i] += realHist.rHist[i];
          sumG[i] += realHist.gHist[i];
          sumB[i] += realHist.bHist[i];
        }
        histogramCount += 1;

        if (oldDNA?.avg_color_stats?.r) processedImage = await colorTransfer(processedImage, oldDNA.avg_color_stats);

        const vision = await analyzeImageWithVision(processedImage);
        if (!vision) continue;

        const typo = inferTypographyStats(vision);
        totals.uppercaseRatio += typo.uppercase_ratio;
        totals.textAreaRatio += typo.text_area_ratio;
        totals.avgWordCount += typo.avg_word_count;
        fontWeights.push(typo.dominant_font_weight);
        textOutlines.push(typo.dominant_text_outline);

        if (vision.face_bbox) {
          faceBBoxTotal.x += vision.face_bbox.x;
          faceBBoxTotal.y += vision.face_bbox.y;
          faceBBoxTotal.w += vision.face_bbox.w;
          faceBBoxTotal.h += vision.face_bbox.h;
        }
        if (vision.text_bbox) {
          textBBoxTotal.x += vision.text_bbox.x;
          textBBoxTotal.y += vision.text_bbox.y;
          textBBoxTotal.w += vision.text_bbox.w;
          textBBoxTotal.h += vision.text_bbox.h;
        }

        addBoxToGrid(occupancyAccum, vision.face_bbox, 1);
        addBoxToGrid(occupancyAccum, vision.text_bbox, 1);

        lightingStyles.push(vision.lighting_style);
        compositionStyles.push(vision.composition_style);
        textPositions.push(vision.text_position);
        artStyles.push(vision.art_style);
        renderTypes.push(vision.render_type);
        distortionTypes.push(vision.subject_distortion_type);
        colorPalettes.push(vision.color_palette_type);

        totals.total += vision.visual_ctr_score || 0;
        totals.face += vision.face_score || 0;
        totals.emotion += vision.emotion_score || 0;
        totals.color += vision.color_score || 0;
        totals.readability += vision.readability_score || 0;
        totals.faceRatio += vision.face_ratio || 0;
        totals.subjectPos += vision.subject_position_x || 0;
        totals.saturation += vision.color_saturation_level || 0;
        totals.contrast += vision.contrast_strength || 0;
        totals.emotionIntensity += vision.emotion_intensity_avg || 0;
        totals.blur += vision.background_blur_level || 0;
        totals.objectDensity += vision.object_density || 0;
        totals.brightness += vision.brightness_mean || 0;
        totals.saturationMean += vision.saturation_mean || 0;
        totals.realism += vision.realism_level || 0;
        totals.negativeSpace += vision.negative_space_ratio || 0;
        totals.horizon += vision.horizon_line_estimate || 0;
        totals.cameraDistance += vision.camera_distance_estimate || 0;

        if (vision.avg_rgb) {
          avgRGBTotal[0] += vision.avg_rgb[0] || 0;
          avgRGBTotal[1] += vision.avg_rgb[1] || 0;
          avgRGBTotal[2] += vision.avg_rgb[2] || 0;
        }

        analyzed.push({ videoId: video.videoId, ...vision });
      } catch (err) {
        console.log("Skip error:", err.message);
      }
    }

    const count = analyzed.length;
    if (!count) return res.json({ error: "No thumbnails analyzed" });

    const normalize = (arr) => {
      const sum = arr.reduce((a, b) => a + b, 0);
      if (sum === 0) return new Array(arr.length).fill(1 / arr.length);
      return arr.map((v) => v / sum);
    };

    const avgR = sumR.map((v) => v / Math.max(histogramCount, 1));
    const avgG = sumG.map((v) => v / Math.max(histogramCount, 1));
    const avgB = sumB.map((v) => v / Math.max(histogramCount, 1));
    const avgColorStats = {
      r: { std: 40, mean: avgRGBTotal[0] / count },
      g: { std: 40, mean: avgRGBTotal[1] / count },
      b: { std: 40, mean: avgRGBTotal[2] / count },
    };

    const subjectDensityMap = normalizeGrid(occupancyAccum, count);
    const negativeSpaceMap = subjectDensityMap.map((row) => row.map((v) => Number((1 - v).toFixed(4))));

    const dna = {
      channel_id: channelId,
      niche_profile: nicheProfile,
      sample_size: count,
      analyzed_at: new Date().toISOString(),
      avg_color_stats: avgColorStats,
      avg_color_histogram_256: { rHist: normalize(avgR), gHist: normalize(avgG), bHist: normalize(avgB) },
      face_bbox_avg: {
        x: Number((faceBBoxTotal.x / count).toFixed(3)),
        y: Number((faceBBoxTotal.y / count).toFixed(3)),
        w: Number((faceBBoxTotal.w / count).toFixed(3)),
        h: Number((faceBBoxTotal.h / count).toFixed(3)),
      },
      text_bbox_avg: {
        x: Number((textBBoxTotal.x / count).toFixed(3)),
        y: Number((textBBoxTotal.y / count).toFixed(3)),
        w: Number((textBBoxTotal.w / count).toFixed(3)),
        h: Number((textBBoxTotal.h / count).toFixed(3)),
      },
      subject_density_map: subjectDensityMap,
      negative_space_map: negativeSpaceMap,
      face_ratio: Number((totals.faceRatio / count).toFixed(3)),
      subject_position_x: Number((totals.subjectPos / count).toFixed(3)),
      color_saturation_level: Number((totals.saturation / count).toFixed(2)),
      contrast_strength: Number((totals.contrast / count).toFixed(2)),
      emotion_intensity_avg: Number((totals.emotionIntensity / count).toFixed(2)),
      background_blur_level: Number((totals.blur / count).toFixed(2)),
      object_density: Number((totals.objectDensity / count).toFixed(2)),
      avg_rgb: [Math.round(avgRGBTotal[0] / count), Math.round(avgRGBTotal[1] / count), Math.round(avgRGBTotal[2] / count)],
      brightness_mean: Number((totals.brightness / count).toFixed(2)),
      saturation_mean: Number((totals.saturationMean / count).toFixed(2)),
      negative_space_ratio: Number((totals.negativeSpace / count).toFixed(2)),
      horizon_line_estimate: Number((totals.horizon / count).toFixed(2)),
      camera_distance_estimate: Number((totals.cameraDistance / count).toFixed(2)),
      dominant_art_style: mostCommon(artStyles),
      dominant_render_type: mostCommon(renderTypes),
      dominant_distortion: mostCommon(distortionTypes),
      dominant_color_palette: mostCommon(colorPalettes),
      dominant_lighting_style: mostCommon(lightingStyles),
      dominant_composition_style: mostCommon(compositionStyles),
      dominant_text_position: mostCommon(textPositions),
      avg_visual_ctr: Math.round(totals.total / count),
      avg_face_score: Math.round(totals.face / count),
      avg_emotion_score: Math.round(totals.emotion / count),
      avg_color_score: Math.round(totals.color / count),
      avg_readability_score: Math.round(totals.readability / count),
      avg_realism_level: Math.round(totals.realism / count),
      uppercase_ratio: Number((totals.uppercaseRatio / count).toFixed(3)),
      dominant_text_outline: mostCommon(textOutlines) === "1" || mostCommon(textOutlines) === 1 ? 1 : 0,
      dominant_font_weight: mostCommon(fontWeights) || "bold",
      text_area_ratio: Number((totals.textAreaRatio / count).toFixed(3)),
      avg_word_count: Number((totals.avgWordCount / count).toFixed(3)),
    };

    saveChannelDNA(channelId, dna);

    // optional controlnet mask
    if (String(req.query.controlnet_mask || "0") === "1") {
      const maskPath = path.join(TEMP_DIR, `mask_${channelId}.png`);
      await createControlNetMaskFromDensityMap(dna.subject_density_map, maskPath);
      dna.controlnet_mask_path = maskPath;
    }

    analyzeCache.set(cacheKey, { ts: Date.now(), data: dna });
    return res.json({ status: "dna_created", dna });
  } catch (err) {
    console.error(err);
    return res.json({ error: err.message });
  }
});

app.get("/generate-thumbnail", async (req, res) => {
  await comfySemaphore.acquire();
  try {
    const videoId = req.query.video_id;
    const version = req.query.workflow_version || "thumbmagic_core_v1.json";
    if (!videoId) return res.json({ error: "video_id is required" });

    const baseWorkflow = safeJsonParse(fs.readFileSync(`./workflows/${version}`, "utf-8"), null);
    if (!baseWorkflow) return res.json({ error: "invalid workflow json" });

    const youtubeData = await getYouTubeData(videoId);
    const channelId = await getChannelIdFromVideo(videoId);
    const channelDNA = loadChannelDNA(channelId);
    const analysis = await analyzeContent(youtubeData, channelDNA);
    if (!analysis) return res.json({ error: "Content analysis failed" });

    const textOptions = Array.isArray(analysis.text_variations) && analysis.text_variations.length
      ? analysis.text_variations
      : [analysis.text_hook || "Must Watch"];

    const negativePrompt = buildNegativePrompt();
    const variations = [];

    for (let i = 0; i < Math.min(5, textOptions.length); i += 1) {
      const workflow = JSON.parse(JSON.stringify(baseWorkflow));
      const finalPrompt = buildPrompt({
        template: analysis.template || "story_cards",
        category: analysis.category || "general",
        focus: analysis.focus || youtubeData.title,
        mood: analysis.mood || "intense",
        text_hook: textOptions[i],
        dna: channelDNA,
      });

      if (DEBUG) {
        console.log("==== FINAL PROMPT ====");
        console.log(finalPrompt);
      }

      const kSamplerNode = findNodeByType(workflow, "KSampler");
      const { positiveNode, negativeNode } = resolvePromptNodes(workflow);
      if (!positiveNode || !negativeNode) return res.json({ error: "Workflow must contain 2 CLIPTextEncode nodes (positive + negative)" });

      workflow[positiveNode].inputs.text = finalPrompt;
      workflow[negativeNode].inputs.text = negativePrompt;
      if (kSamplerNode) workflow[kSamplerNode].inputs.seed = Math.floor(Math.random() * 1000000000);

      if (channelDNA?.subject_density_map && String(req.query.controlnet_mask || "0") === "1") {
        const maskPath = path.join(TEMP_DIR, `gen_mask_${channelId}_${Date.now()}_${i}.png`);
        await createControlNetMaskFromDensityMap(channelDNA.subject_density_map, maskPath);
        // optional injection when workflow has image loader node expecting mask path
        const maskLoaderNode = findNodeByType(workflow, "LoadImage");
        if (maskLoaderNode && workflow[maskLoaderNode]?.inputs?.image !== undefined) {
          workflow[maskLoaderNode].inputs.image = maskPath;
        }
      }

      const comfyResponse = await axios.post(`${COMFY_URL}/prompt`, { prompt: workflow });
      const promptId = comfyResponse.data.prompt_id;
      const imageFilename = await waitForComfyImage(promptId);
      if (!imageFilename) continue;

      const imageUrl = `${COMFY_URL}/view?filename=${imageFilename}`;
      const imageResponse = await axios.get(imageUrl, { responseType: "arraybuffer" });
      let processedImage = Buffer.from(imageResponse.data);

      if (channelDNA?.avg_color_histogram_256?.rHist?.length === 256) {
        processedImage = await matchHistogram(processedImage, channelDNA.avg_color_histogram_256);
      }
      if (channelDNA?.avg_color_stats?.r) {
        processedImage = await colorTransfer(processedImage, channelDNA.avg_color_stats, 0.4);
      }

      const visionAnalysis = await analyzeImageWithVision(processedImage);
      if (!visionAnalysis) continue;

      const overlap = bboxOverlapRatio(normalizeBBox(visionAnalysis.face_bbox), normalizeBBox(visionAnalysis.text_bbox));
      if (overlap > 0.15) {
        // regenerate policy: skip this variation if text overlaps face too much
        continue;
      }

      variations.push({
        text: textOptions[i],
        image: imageUrl,
        base_score: calculateFinalScore(analysis),
        vision_score: visionAnalysis.visual_ctr_score,
        face_score: visionAnalysis.face_score,
        emotion_score: visionAnalysis.emotion_score,
        color_score: visionAnalysis.color_score,
        readability_score: visionAnalysis.readability_score,
        subject_position_x: clip01(visionAnalysis.subject_position_x),
        negative_space_ratio: clip01(visionAnalysis.negative_space_ratio),
      });
    }

    let winner = null;
    if (variations.length > 0) {
      winner = variations.reduce((best, current) => (current.vision_score > best.vision_score ? current : best));
      if (winner && channelDNA) updateDNAFromWinner(channelId, winner, channelDNA);
    }

    return res.json({
      status: "completed",
      title: youtubeData.title,
      overall_score: calculateFinalScore(analysis),
      ctr_score: analysis.ctr_score,
      hook_strength: analysis.hook_strength,
      overall_text_score: calculateFinalScore(analysis),
      emotion_intensity: analysis.emotion_intensity,
      winner,
      thumbnails: variations,
      concurrency_limit: 2,
    });
  } catch (err) {
    console.error(err);
    return res.json({ error: err.message });
  } finally {
    comfySemaphore.release();
  }
});

app.listen(9000, () => {
  console.log("Server running on http://localhost:9000");
});
