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
const OpenAI = require("openai");
const { matchHistogram, getHistogram } = require("./coloranalysis");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const app = express();
app.use(express.json());

const COMFY_URL = "http://127.0.0.1:8188";
const YOUTUBE_API_KEY = process.env.YOUTUBE_API_KEY;
const path = require("path");
const DNA_DIR = path.join(__dirname, "channel_dna");

if (!fs.existsSync(DNA_DIR)) {
  fs.mkdirSync(DNA_DIR);
}
async function getMeanStd(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .raw()
    .toBuffer({ resolveWithObject: true });

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

  return {
    r: calc(stats.r),
    g: calc(stats.g),
    b: calc(stats.b),
  };
}

async function colorTransfer(imageBuffer, targetStats, weight = 0.4) {

  const source = await getMeanStd(imageBuffer);

  const safe = (v) => (v === 0 ? 1 : v);

  const scaleR = 1 + weight * ((targetStats.r.std / safe(source.r.std)) - 1);
  const scaleG = 1 + weight * ((targetStats.g.std / safe(source.g.std)) - 1);
  const scaleB = 1 + weight * ((targetStats.b.std / safe(source.b.std)) - 1);

  const shiftR = weight * (targetStats.r.mean - source.r.mean);
  const shiftG = weight * (targetStats.g.mean - source.g.mean);
  const shiftB = weight * (targetStats.b.mean - source.b.mean);

  return sharp(imageBuffer)
    .linear(
      [scaleR, scaleG, scaleB],
      [shiftR, shiftG, shiftB]
    )
    .toBuffer();
}
/* =========================
   📁 Ensure temp folder exists
========================= */
const TEMP_DIR = path.join(__dirname, "temp");
if (!fs.existsSync(TEMP_DIR)) {
  fs.mkdirSync(TEMP_DIR);
}
/* =========================
   🔥 LOAD CHANNEL DNA
========================= */
function mostCommon(arr) {
  if (!arr || arr.length === 0) return null;

  const counts = {};

  arr.forEach(item => {
    if (!item) return;
    counts[item] = (counts[item] || 0) + 1;
  });

  let max = 0;
  let result = null;

  for (let key in counts) {
    if (counts[key] > max) {
      max = counts[key];
      result = key;
    }
  }

  return result;
}
function loadChannelDNA(channelId) {
  const dnaPath = path.join(__dirname, "channel_dna", `${channelId}.json`);

  if (!fs.existsSync(dnaPath)) {
    return null;
  }

  const raw = fs.readFileSync(dnaPath, "utf-8");
  return JSON.parse(raw);
}
function updateDNAFromWinner(channelId, winner, channelDNA) {

  if (!channelDNA || !winner) return;

  const dnaPath = path.join(DNA_DIR, `${channelId}.json`);

  // 1️⃣ Update avg_visual_ctr
  const newCTR = Math.round(
    (channelDNA.avg_visual_ctr * channelDNA.sample_size + winner.vision_score)
    / (channelDNA.sample_size + 1)
  );

  channelDNA.avg_visual_ctr = newCTR;
  channelDNA.sample_size += 1;

  // 2️⃣ Boost readability if winner strong
  if (winner.readability_score > channelDNA.avg_readability_score) {
    channelDNA.avg_readability_score =
      Math.round((channelDNA.avg_readability_score + winner.readability_score) / 2);
  }

  // 3️⃣ Boost emotion weight
  if (winner.emotion_score > channelDNA.avg_emotion_score) {
    channelDNA.avg_emotion_score =
      Math.round((channelDNA.avg_emotion_score + winner.emotion_score) / 2);
  }

  // 4️⃣ Save back to file
  fs.writeFileSync(dnaPath, JSON.stringify(channelDNA, null, 2));

  console.log("🔥 DNA updated from winner");
}
/* =========================
   🔥 STEP 1 — GET CHANNEL VIDEOS
========================= */

async function getChannelVideos(channelId) {
  const response = await axios.get(
    "https://www.googleapis.com/youtube/v3/search",
    {
      params: {
        part: "snippet",
        channelId: channelId,
        maxResults: 20,
        order: "date",
        type: "video",
        key: YOUTUBE_API_KEY,
      },
    }
  );

  return response.data.items.map(item => ({
    videoId: item.id.videoId,
    thumbnail: item.snippet.thumbnails.high.url
  }));
}

/* =========================
   🔥 GET VIDEO DETAILS (for GPT niche analysis)
========================= */

async function getChannelVideoDetails(channelId) {

  const search = await axios.get(
    "https://www.googleapis.com/youtube/v3/search",
    {
      params: {
        part: "snippet",
        channelId,
        maxResults: 15,
        order: "date",
        type: "video",
        key: YOUTUBE_API_KEY
      }
    }
  );

  const videoIds = search.data.items.map(v => v.id.videoId).join(",");

  const videos = await axios.get(
    "https://www.googleapis.com/youtube/v3/videos",
    {
      params: {
        part: "snippet,statistics",
        id: videoIds,
        key: YOUTUBE_API_KEY
      }
    }
  );

  return videos.data.items.map(v => ({
    title: v.snippet.title,
    description: v.snippet.description,
    tags: v.snippet.tags || [],
    categoryId: v.snippet.categoryId,
    viewCount: v.statistics.viewCount,
    likeCount: v.statistics.likeCount
  }));
}

/* =========================
   🔥 ANALYZE NICHE WITH GPT
========================= */

async function analyzeChannelNiche(videoList) {

  const combinedText = videoList.map(v =>
    `Title: ${v.title}\nDescription: ${v.description}\nTags: ${v.tags.join(", ")}`
  ).join("\n\n");

  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: "You are a YouTube channel strategist. Analyze channel niche deeply."
      },
      {
        role: "user",
        content: `
Analyze this channel content and return JSON:

{
  "niche": "",
  "primary_topic": "",
  "common_objects": [],
  "emotion_baseline": "",
  "style_profile": "",
  "visual_theme": ""
}

Channel Data:
${combinedText}
`
      }
    ]
  });

  return JSON.parse(response.choices[0].message.content);
}
 


/* =========================
   🔥 ANALYZE CHANNEL API
========================= */
/* =========================
   🧬 ANALYZE CHANNEL → BUILD DNA
========================= */
app.get("/analyze-channel", async (req, res) => {
  try {
    const channelId = req.query.channel_id;

    if (!channelId) {
      return res.json({ error: "channel_id is required" });
    }
  // 1️⃣ ดึง metadata
    const videoDetails = await getChannelVideoDetails(channelId);

    // 2️⃣ วิเคราะห์ niche ด้วย GPT
    const nicheProfile = await analyzeChannelNiche(videoDetails);

    const videos = await getChannelVideos(channelId);

    if (!videos.length) {
      return res.json({ error: "No videos found" });
    }
const channelDNA = loadChannelDNA(channelId);

let rStdTotal = 0;
let gStdTotal = 0;
let bStdTotal = 0;

    let total = 0;
    let faceTotal = 0;
    let emotionTotal = 0;
    let colorTotal = 0;
    let readabilityTotal = 0;
    
    let lightingStyles = [];
    let compositionStyles = [];
    let textPositions = [];
    let realismTotal = 0;
let faceRatioTotal = 0;
let subjectPosTotal = 0;
let saturationTotal = 0;
let contrastTotal = 0;
let emotionIntensityTotal = 0;
let blurTotal = 0;
let objectDensityTotal = 0;
let brightnessTotal = 0;
let saturationMeanTotal = 0;

let avgRGBTotal = [0, 0, 0];
let histogramTotal = [];
// 🔥 REAL 256-bin histogram accumulate
let sumR = new Array(256).fill(0);
let sumG = new Array(256).fill(0);
let sumB = new Array(256).fill(0);
let histogramCount = 0;
let contrastCurveTotal = [];
let negativeSpaceTotal = 0;
let horizonTotal = 0;
let cameraDistanceTotal = 0;

// 🔥 BBOX AGGREGATION (Phase 1 complete)
let faceBBoxTotal = { x: 0, y: 0, w: 0, h: 0 };
let textBBoxTotal = { x: 0, y: 0, w: 0, h: 0 };
    let artStyles = [];

let renderTypes = [];
let distortionTypes = [];
let colorPalettes = [];

    const analyzed = [];

    for (let video of videos) {
      try {
       const imageUrl = video.thumbnail;

const imageResponse = await axios.get(imageUrl, {
  responseType: "arraybuffer"
});

const imageBuffer = Buffer.from(imageResponse.data);
// 🔥 REAL histogram from pixels (NOT GPT)
const realHist = await getHistogram(imageBuffer);

for (let i = 0; i < 256; i++) {
  sumR[i] += realHist.rHist[i];
  sumG[i] += realHist.gHist[i];
  sumB[i] += realHist.bHist[i];
}

histogramCount++;

let processedImage = imageBuffer;

if (channelDNA && channelDNA.sample_size > 5) {
  processedImage = await colorTransfer(
    processedImage,
    channelDNA.avg_color_stats
  );
}
        const vision = await analyzeImageWithVision(processedImage);
if (!vision) continue;

if (vision.color_stats) {
  rStdTotal += vision.color_stats.r.std;
  gStdTotal += vision.color_stats.g.std;
  bStdTotal += vision.color_stats.b.std;
}
// 🔥 accumulate face bbox
if (vision.face_bbox) {
  faceBBoxTotal.x += vision.face_bbox.x;
  faceBBoxTotal.y += vision.face_bbox.y;
  faceBBoxTotal.w += vision.face_bbox.w;
  faceBBoxTotal.h += vision.face_bbox.h;
}

// 🔥 accumulate text bbox
if (vision.text_bbox) {
  textBBoxTotal.x += vision.text_bbox.x;
  textBBoxTotal.y += vision.text_bbox.y;
  textBBoxTotal.w += vision.text_bbox.w;
  textBBoxTotal.h += vision.text_bbox.h;
}
        lightingStyles.push(vision.lighting_style);
compositionStyles.push(vision.composition_style);
textPositions.push(vision.text_position);
realismTotal += vision.realism_level;
artStyles.push(vision.art_style);
renderTypes.push(vision.render_type);
distortionTypes.push(vision.subject_distortion_type);
colorPalettes.push(vision.color_palette_type);
       total += vision.visual_ctr_score;
        faceTotal += vision.face_score;
        emotionTotal += vision.emotion_score;
        colorTotal += vision.color_score;
        readabilityTotal += vision.readability_score;
faceRatioTotal += vision.face_ratio || 0;
subjectPosTotal += vision.subject_position_x || 0;
saturationTotal += vision.color_saturation_level || 0;
contrastTotal += vision.contrast_strength || 0;
emotionIntensityTotal += vision.emotion_intensity_avg || 0;
blurTotal += vision.background_blur_level || 0;
objectDensityTotal += vision.object_density || 0;
brightnessTotal += vision.brightness_mean || 0;
saturationMeanTotal += vision.saturation_mean || 0;

if (vision.color_histogram && vision.color_histogram.length) {
  histogramTotal.push(vision.color_histogram);
}

if (vision.contrast_curve && vision.contrast_curve.length) {
  contrastCurveTotal.push(vision.contrast_curve);
}

if (vision.avg_rgb) {
  avgRGBTotal[0] += vision.avg_rgb[0];
  avgRGBTotal[1] += vision.avg_rgb[1];
  avgRGBTotal[2] += vision.avg_rgb[2];
}
negativeSpaceTotal += vision.negative_space_ratio || 0;
horizonTotal += vision.horizon_line_estimate || 0;
cameraDistanceTotal += vision.camera_distance_estimate || 0;
        analyzed.push({
          videoId: video.videoId,
          ...vision
        });

        console.log("Analyzed:", video.videoId);

      } catch (err) {
        console.log("Skip error:", err.message);
      }
    }

    const count = analyzed.length;
    // =============================
// 🔥 NORMALIZE HISTOGRAM
// =============================
// =============================
// 🔥 NORMALIZE HISTOGRAM
// =============================
let normalizedHistogram = [];

if (histogramTotal.length) {

  // 🔥 หา length ที่สั้นที่สุด
  const minLength = Math.min(
    ...histogramTotal.map(h => h.length)
  );

  // 🔥 ตัดทุก histogram ให้เท่ากันก่อน
  const trimmed = histogramTotal.map(h =>
    h.slice(0, minLength)
  );

  const avgHistogram = trimmed[0].map((_, i) =>
    trimmed.reduce((sum, arr) => sum + (arr[i] || 0), 0) /
    trimmed.length
  );

  const sumHist = avgHistogram.reduce((a, b) => a + b, 0);

  if (sumHist > 0) {
    normalizedHistogram = avgHistogram.map(v =>
      Number((v / sumHist).toFixed(4))
    );
  }
}

// =============================
// 🔥 NORMALIZE CONTRAST CURVE
// =============================
let normalizedContrast = [];

if (contrastCurveTotal.length) {

  const avgContrast = contrastCurveTotal[0].map((_, i) =>
    contrastCurveTotal.reduce((sum, arr) => sum + (arr[i] || 0), 0) /
    contrastCurveTotal.length
  );

  const sumContrast = avgContrast.reduce((a, b) => a + b, 0);

  normalizedContrast = avgContrast.map(v =>
    Number((v / sumContrast).toFixed(4))
  );
}

    if (!count) {
      return res.json({ error: "No thumbnails analyzed" });
    }
// 🔥 Normalize REAL histogram
// 🔥 NORMALIZE REAL 256 HISTOGRAM (ALWAYS SAFE)
let normalizedHist256 = {
  rHist: new Array(256).fill(1 / 256),
  gHist: new Array(256).fill(1 / 256),
  bHist: new Array(256).fill(1 / 256),
};

if (histogramCount > 0) {

  const avgR = sumR.map(v => v / histogramCount);
  const avgG = sumG.map(v => v / histogramCount);
  const avgB = sumB.map(v => v / histogramCount);

  const normalize = (arr) => {
    const sum = arr.reduce((a, b) => a + b, 0);
    if (sum === 0) return new Array(256).fill(1 / 256);
    return arr.map(v => v / sum);
  };

  normalizedHist256 = {
    rHist: normalize(avgR),
    gHist: normalize(avgG),
    bHist: normalize(avgB),
  };
}
    const dna = {
      
      avg_color_histogram_256: {
  rHist: normalizedHist256.rHist,
  gHist: normalizedHist256.gHist,
  bHist: normalizedHist256.bHist
}
,
      face_bbox_avg: {
  x: Number((faceBBoxTotal.x / count).toFixed(2)),
  y: Number((faceBBoxTotal.y / count).toFixed(2)),
  w: Number((faceBBoxTotal.w / count).toFixed(2)),
  h: Number((faceBBoxTotal.h / count).toFixed(2)),
},

text_bbox_avg: {
  x: Number((textBBoxTotal.x / count).toFixed(2)),
  y: Number((textBBoxTotal.y / count).toFixed(2)),
  w: Number((textBBoxTotal.w / count).toFixed(2)),
  h: Number((textBBoxTotal.h / count).toFixed(2)),
},

avg_color_histogram: normalizedHistogram,
avg_contrast_curve: normalizedContrast,

      // 🧬 VECTOR STYLE FINGERPRINT
face_ratio: Number((faceRatioTotal / count).toFixed(2)),
subject_position_x: Number((subjectPosTotal / count).toFixed(2)),
color_saturation_level: Number((saturationTotal / count).toFixed(2)),
contrast_strength: Number((contrastTotal / count).toFixed(2)),
emotion_intensity_avg: Number((emotionIntensityTotal / count).toFixed(2)),
background_blur_level: Number((blurTotal / count).toFixed(2)),
object_density: Number((objectDensityTotal / count).toFixed(2)),
avg_rgb: [
  Math.round(avgRGBTotal[0] / count),
  Math.round(avgRGBTotal[1] / count),
  Math.round(avgRGBTotal[2] / count),
],



brightness_mean: Number((brightnessTotal / count).toFixed(2)),
saturation_mean: Number((saturationMeanTotal / count).toFixed(2)),
negative_space_ratio: Number((negativeSpaceTotal / count).toFixed(2)),
horizon_line_estimate: Number((horizonTotal / count).toFixed(2)),
camera_distance_estimate: Number((cameraDistanceTotal / count).toFixed(2)),
      dominant_art_style: mostCommon(artStyles),
dominant_render_type: mostCommon(renderTypes),
dominant_distortion: mostCommon(distortionTypes),
dominant_color_palette: mostCommon(colorPalettes),
  dominant_color_histogram: normalizedHistogram,
dominant_contrast_curve: normalizedContrast,
  channel_id: channelId,
  niche_profile: nicheProfile,
  sample_size: count,
  avg_visual_ctr: Math.round(total / count),
  avg_face_score: Math.round(faceTotal / count),
  avg_emotion_score: Math.round(emotionTotal / count),
  avg_color_score: Math.round(colorTotal / count),
  avg_readability_score: Math.round(readabilityTotal / count),
  analyzed_at: new Date().toISOString(),

  dominant_lighting_style: mostCommon(lightingStyles),
  dominant_composition_style: mostCommon(compositionStyles),
  dominant_text_position: mostCommon(textPositions),
  avg_realism_level: Math.round(realismTotal / count)
};

    const dnaPath = path.join(__dirname, "channel_dna", `${channelId}.json`);
    fs.writeFileSync(dnaPath, JSON.stringify(dna, null, 2));

    res.json({
      status: "dna_created",
      dna
    });

    } catch (err) {
    console.error(err);
    res.json({ error: err.message });
  }
});
/* =========================
   🔥 GPT ANALYSIS
========================= */
async function analyzeImageWithVision(imageBuffer) {
  const base64Image = imageBuffer.toString("base64");

  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    response_format: { type: "json_object" },
    messages: [
      {
  role: "system",
  content: `
You are a YouTube thumbnail CTR expert.

IMPORTANT RULE:
All ratio values MUST be between 0.0 and 1.0 only.
Never use 0-100 scale for ratios.

All bbox values (x,y,w,h) MUST be between 0.0 and 1.0.
Coordinates are normalized relative to full image.

0.0 = none / far left / no space
1.0 = full frame / far right / maximum

CTR scores still use 1-100 scale.

Return JSON only.
`
},
      {
        role: "user",
        content: [
          {
            type: "text",
            text: `
Score this thumbnail based on:
- Face size dominance
- Emotional clarity
- Color contrast
- Subject separation
- Mobile readability

Also identify:
- art style (photorealistic, anime, cartoon, 3D render, stylized digital painting)
- render type (AI photo, 3D game render, painted illustration)
- character exaggeration level (0-100)
- background complexity (0-100)
- color palette type (warm cinematic, neon cyberpunk, pastel, dark moody)
- outline presence (strong, soft, none)
- subject distortion type (big head small body, normal proportion, hyper realistic)

Return JSON:

{
  "visual_ctr_score": 0,
  "face_score": 0,
  "emotion_score": 0,
  "color_score": 0,
  "readability_score": 0,

   // All ratio values MUST be 0.0 - 1.0
  "face_ratio": 0.0,
  "subject_position_x": 0.0,
  "negative_space_ratio": 0.0,
  "camera_distance_estimate": 0.0,
  "horizon_line_estimate": 0.0,

  "color_saturation_level": 0,
  "contrast_strength": 0,
  "emotion_intensity_avg": 0,
  "background_blur_level": 0,
  "object_density": 0,

  "lighting_style": "",
  "composition_style": "",
  "text_position": "",
  "realism_level": 0,

  "art_style": "",
  "render_type": "",
  "subject_distortion_type": "",
  "color_palette_type": "",

  "face_bbox": {
    "x": 0,
    "y": 0,
    "w": 0,
    "h": 0
  },

  "text_bbox": {
    "x": 0,
    "y": 0,
    "w": 0,
    "h": 0
  },

    "color_stats": {
    "r": { "mean": 0, "std": 0 },
    "g": { "mean": 0, "std": 0 },
    "b": { "mean": 0, "std": 0 }
  },

  "avg_rgb": [0, 0, 0],
  "brightness_mean": 0,
  "saturation_mean": 0,
  "color_histogram": [],
  "contrast_curve": []
}
`
          },
          {
            type: "image_url",
            image_url: {
              url: `data:image/png;base64,${base64Image}`
            }
          }
        ]
      }
    ]
  });

  let content = response.choices[0].message.content;

try {
  const parsed = JSON.parse(content);
function normalizeBBox(bbox) {
  if (!bbox) return { x: 0, y: 0, w: 0, h: 0 };

  function fix(v) {
    if (typeof v !== "number") return 0;
    if (v > 1) v = v / 100;
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    return Number(v.toFixed(2));
  }

  return {
    x: fix(bbox.x),
    y: fix(bbox.y),
    w: fix(bbox.w),
    h: fix(bbox.h),
  };
}

parsed.face_bbox = normalizeBBox(parsed.face_bbox);
parsed.text_bbox = normalizeBBox(parsed.text_bbox);

  function normalizeRatio(value) {
  if (!value) return 0;

  if (value > 1.5) value = value / 100;

  if (value > 1) value = 1;

  if (value < 0) value = 0;

  return Number(value.toFixed(3));
}

  parsed.face_ratio = normalizeRatio(parsed.face_ratio);
  parsed.subject_position_x = normalizeRatio(parsed.subject_position_x);
  parsed.negative_space_ratio = normalizeRatio(parsed.negative_space_ratio);
  parsed.camera_distance_estimate = normalizeRatio(parsed.camera_distance_estimate);
  parsed.horizon_line_estimate = normalizeRatio(parsed.horizon_line_estimate);

  return parsed;

} catch (err) {
  console.error("Vision JSON parse error:", err.message);
  console.error("Raw response was:", content);
  return null;
}
}


async function analyzeContent(data, channelDNA) {

  let dnaBlock = "";

  if (channelDNA) {
    dnaBlock = `
Channel Style Baseline:
- Dominant lighting: ${channelDNA.dominant_lighting_style}
- Dominant composition: ${channelDNA.dominant_composition_style}
- Dominant text position: ${channelDNA.dominant_text_position}
- Emotional baseline: ${channelDNA.niche_profile?.emotion_baseline}
- Visual theme: ${channelDNA.niche_profile?.visual_theme}

When generating analysis:
Adapt to this channel identity.
Do NOT invent random aesthetics outside this baseline.
`;
  }

  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `
You are an elite YouTube CTR strategist.

${dnaBlock}

Analyze viral psychology deeply.

If the video is about a specific game (ROV, Minecraft, Valorant, etc),
you MUST extract specific in-game objects, characters, abilities, weapons, UI elements, maps, or recognizable game visuals.

Never stay generic. Be concrete and visual.

Always return valid JSON only.
`
      },
      {
  role: "user",
  content: `
Analyze this YouTube video data and return ONLY JSON.

Video topic: ${data.title}
Description: ${data.description}

Extract:
- main subject focus
- emotional trigger
- key visual objects
- visual tension level

Scoring rules:
- hook_strength (1-10)
- emotion_intensity (1-10)
- ctr_score (1-100)

Generate 3 short powerful text variations (2-4 words each).

Return:

{
  "category": "",
  "template": "",
  "game": "",
  "specific_game_elements": [],
  "visual_objects": [],
  "environment_type": "",
  "mood": "",
  "focus": "",
  "style": "",
  "layout": "",
  "text_hook": "",
  "hook_strength": 0,
  "emotion_intensity": 0,
  "ctr_score": 0,
  "text_variations": []
}

Video Data:
Title: ${data.title}
Description: ${data.description}
Tags: ${data.tags.join(", ")}
YouTube Category ID: ${data.categoryId}
Views: ${data.viewCount}
Likes: ${data.likeCount}
`
      }
    ]
  });

  try {
    return JSON.parse(response.choices[0].message.content);
  } catch (err) {
    console.error("Content JSON parse error:", err.message);
    return null;
  }
}

/* =========================
   🔥 TEMPLATE ENGINE
========================= */
function applyTemplateLogic(context) {
  switch (context.template) {
    case "story_cards":
      return `
Cinematic storytelling frame,
Character reacting emotionally,
Background supports story tension,
`;

    case "text_on_image":
      return `
Large central bold text,
Clean background,
Face positioned side,
`;

    case "floating_proof":
      return `
Money or numbers floating,
Glow effect,
Subject reacting,
`;

    case "before_after":
      return `
Split screen layout,
Clear BEFORE vs AFTER contrast,
`;

    case "podcast":
      return `
Two faces conversation,
Microphone visible,
`;

    case "object_in_hand":
      return `
Subject holding object,
Object emphasized clearly,
`;

    case "contextual_background":
      return `
Large dramatic environment,
Environmental storytelling,
`;

    case "explainer":
      return `
Subject pointing at UI graphics,
Educational vibe,
`;

    default:
      return "";
  }
}
// ===============================
// 1️⃣ Blueprint Generator Layer
// ===============================

async function generateVisualBlueprint(videoData, dna) {
  const response = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content: `
You are a visual director for high CTR YouTube thumbnails.
Generate a strict JSON blueprint for thumbnail composition.
Return only JSON.
`
      },
      {
        role: "user",
        content: `
Video topic: ${videoData.topic}
Emotion: ${videoData.emotion}
Main subject focus: ${videoData.focus}
Key objects: ${videoData.objects?.join(", ")}

Channel niche: ${dna?.niche_profile?.niche}
Dominant composition: ${dna?.dominant_composition_style}
Dominant text position: ${dna?.dominant_text_position}

Return JSON with this structure:

{
  framing: "",
  subject_type: "",
  camera_angle: "",
  emotion_type: "",
  lighting_type: "",
  background_type: "",
  text_zone: "",
  visual_priority_order: []
}
`
      }
    ]
  });

  try {
  return JSON.parse(response.choices[0].message.content);
} catch (err) {
  console.error("Blueprint JSON parse error:", err.message);
  console.error("Raw blueprint response:", response.choices[0].message.content);
  return null;
}
}
/* =========================
   🔥 BUILD PROMPT
========================= */

function buildPrompt(context) {
   let structuralInjection = "";
   
  if (context.dna?.avg_color_stats?.r) {
  structuralInjection += `
Maintain color variance similar to channel:
Red variance ${context.dna.avg_color_stats.r.std},
Green variance ${context.dna.avg_color_stats.g.std},
Blue variance ${context.dna.avg_color_stats.b.std},
`;
}
  const templateInstruction = applyTemplateLogic(context);
  // ===============================
// 🔥 CTR ADAPTIVE TOKEN SCALING
// ===============================

let exaggerationWeight = 1.0;
let emotionWeight = 1.0;
let colorWeight = 1.0;

if (context.dna) {

  // 🔥 Visual CTR scaling
  if (context.dna?.avg_visual_ctr > 85) {
  exaggerationWeight = 1.5;
  emotionWeight = 1.4;
  colorWeight = 1.3;
}
else if (context.dna?.avg_visual_ctr > 75) {
  exaggerationWeight = 1.2;
  emotionWeight = 1.2;
  colorWeight = 1.1;
}
else if (context.dna?.avg_visual_ctr < 60) {
  exaggerationWeight = 0.9;
  emotionWeight = 0.9;
  colorWeight = 0.95;
}

  // 🔥 Emotion scaling
  if (context.dna.avg_emotion_score > 75) {
    emotionWeight = 1.4;
  }

  // 🔥 Color scaling
  if (context.dna.avg_color_score > 75) {
    colorWeight = 1.4;
  }
}
  let blueprintBlock = "";

  

if (context.dna?.avg_contrast_curve?.length) {
  structuralInjection += `
(contrast curve profile: ${context.dna.avg_contrast_curve.join(", ")}:1.4),
`;
}

if (context.dna?.avg_color_histogram?.length) {
  structuralInjection += `
(color histogram profile: ${context.dna.avg_color_histogram.join(", ")}:1.4),
`;
}
  

if (context.blueprint) {
  blueprintBlock = `
Framing: ${context.blueprint.framing}
Subject type: ${context.blueprint.subject_type}
Camera angle: ${context.blueprint.camera_angle}
Emotion type: ${context.blueprint.emotion_type}
Lighting: ${context.blueprint.lighting_type}
Background: ${context.blueprint.background_type}
Text zone: ${context.blueprint.text_zone}
Visual priority: ${context.blueprint.visual_priority_order.join(", ")}
`;
}
  let videoContextBlock = `
Based on actual YouTube video:
Title: ${context.videoTitle},
Description summary: ${context.videoDescription?.slice(0, 300)},
Core tags: ${context.videoTags?.slice(0, 8).join(", ")},

Visual storytelling must reflect this topic accurately.
`;
let nicheBlock = "";

if (context.dna?.niche_profile) {

  const niche = context.dna.niche_profile;

  nicheBlock = `
Channel Niche Intelligence:
Main niche: ${niche.niche},
Primary topic: ${niche.primary_topic},
Common visual objects: ${niche.common_objects?.join(", ")},
Emotional baseline: ${niche.emotion_baseline},
Visual theme direction: ${niche.visual_theme},

Adapt to this channel identity while improving CTR potential.
`;
}
  // 🧬 DNA LAYOUT OVERRIDE
let dnaLayoutOverride = "";

if (context.dna && context.dna.sample_size >= 8) {

  if (context.dna?.dominant_composition_style?.includes("split")) {
    dnaLayoutOverride += `
Split screen composition,
Clear left vs right separation,
Strong visual contrast between sides,
`;
  }

  if (context.dna.dominant_composition_style?.includes("center")) {
    dnaLayoutOverride += `
Centered face dominance,
Symmetrical composition,
Face placed dead center,
`;
  }

  if (context.dna.dominant_composition_style?.includes("side")) {
    dnaLayoutOverride += `
Subject on one side,
Large empty space for bold text,
Asymmetrical thumbnail layout,
`;
  }

}
  if (!context.dna || context.dna.sample_size < 8) {
  context.dna = null;
}
let dnaBoost = "";

  // 🎨 GLOBAL COLOR LOCK (Always applied)
let colorLock = `
high contrast lighting,
professional youtube thumbnail color grading,
`;

if (context.dna && context.dna.dominant_lighting_style?.includes("soft")) {
  colorLock = `
soft cinematic lighting,
natural color grading,
`;
} else {
  colorLock = `
(high contrast:${colorWeight}),
(neon rim light:${colorWeight}),
(cinematic lighting:1.3),
`;
}
// 🧱 BACKGROUND STRUCTURE LOCK (Always applied)
let backgroundLock = `
foreground subject dominant,
background slightly blurred but colorful,
big shapes,
clean composition,
clear separation,
`;
let styleSignatureBlock = "";

let colorVectorLock = "";


if (context.dna?.avg_rgb) {
  colorVectorLock = `
Average RGB tone: ${context.dna.avg_rgb.join(", ")},
Match overall brightness ${context.dna.brightness_mean},
Match saturation level ${context.dna.saturation_mean},
Keep color grading consistent with channel identity,
`;
}

let geometryControl = "";

if (context.dna?.negative_space_ratio > 0.4) {
  geometryControl += `
Large empty negative space preserved,
`;
}

if (context.dna?.camera_distance_estimate > 0.6) {
  geometryControl += `
Medium camera distance framing,
`;
} else {
  geometryControl += `
Close-up framing,
`;
}
if (context.dna?.dominant_art_style) {
  styleSignatureBlock = `
Art style: ${context.dna.dominant_art_style}
Rendering type: ${context.dna.dominant_render_type}
Color palette direction: ${context.dna.dominant_color_palette}
Character distortion: ${context.dna.dominant_distortion}
`;
}
let vectorControl = "";

if (context.dna?.face_ratio) {
  vectorControl += `
Face occupies ${(context.dna.face_ratio * 100).toFixed(0)}% of frame,
`;
}

if (context.dna?.subject_position_x < 0.4) {
  vectorControl += `
Subject positioned on left side,
`;
}

if (context.dna?.subject_position_x > 0.6) {
  vectorControl += `
Subject positioned on right side,
`;
}

// 🎯 MASTER STYLE CORE (Refined)
let masterStyleCore = "";

if (context.dna && context.dna.avg_face_score < 60) {
  masterStyleCore = `
Waist up framing,
Environment visible,
Less extreme close up,
Balanced composition,
`;
} else {
  masterStyleCore = `

face dominant but not centered,
(strong emotional intensity:${exaggerationWeight}),
`;
}
// 🎯 MICRO DETAIL REFINEMENT
let microRefinement = `
natural skin texture,
realistic facial lighting transitions,
subtle color harmony,
professional youtube thumbnail polish,
balanced highlights and shadows
`;
    // 🔥 STYLE LOCK (Thumbmagic-like)
let styleLock = `
Medium close-up,
face filling 65% of frame,
24mm ultra wide lens distortion,
head near top frame edge,

intense facial expression,
mouth slightly open,
dramatic eyebrows,

hyper saturated colors,
high contrast lighting,
neon rim light,
cinematic thumbnail style,

clean bold background,
colorful environment,
foreground dominance,
sharp focus,
`;
// 🧬 LAYOUT DNA MAPPING (REAL STRUCTURE CONTROL)
let layoutDNA = "";

if (context.dna && context.dna.dominant_composition_style) {

  if (context.dna.dominant_composition_style.includes("side")) {
    layoutDNA = `
    (subject positioned on left third:1.6),
    (large empty negative space on right for text:1.6),
    (asymmetrical composition:1.5),
    (rule of thirds framing:1.4),
    (subject NOT centered:1.6),
    `;
  }

  else if (context.dna.dominant_composition_style.includes("center")) {
    layoutDNA = `
    (strong centered face dominance:1.4),
    (symmetrical composition:1.3),
    (balanced layout:1.2),
    `;
  }

  else if (context.dna.dominant_composition_style.includes("split")) {
    layoutDNA = `
    (split screen composition:1.6),
    (clear left vs right contrast:1.5),
    (strong visual separation:1.4),
    `;
  }
}


  // ===============================
// 🧬 DNA STRUCTURAL LAYOUT ENGINE
// ===============================

let framing = "";
let compositionControl = "";
let textZoneControl = "";

if (context.dna && context.dna.sample_size >= 8) {

  const comp = context.dna.dominant_composition_style || "";
  const textPos = context.dna.dominant_text_position || "";

  // 🎯 COMPOSITION CONTROL
   if (comp.includes("dynamic")) {
  compositionControl = `
Dynamic diagonal composition,
Energy flow across frame,
Asymmetrical balance,
`;
  framing = "Medium close-up dynamic angle,";
}
  if (comp.includes("split")) {
    compositionControl = `
Split screen layout,
Left vs right visual contrast,
Clear visual separation line,
One side face, other side supporting element,
`;
    framing = "Face occupies 45% of frame,";
  }

  else if (comp.includes("side")) {
    compositionControl = `
Subject positioned on one side,
Strong empty negative space on opposite side,
Asymmetrical layout,
`;
    framing = "Face filling 50% of frame, offset to side,";
  }

  else {
    compositionControl = `
Centered dominant subject,
Symmetrical composition,
`;
    framing = "Extreme close-up face filling 70% of frame,";
  }

  // 📝 TEXT ZONE CONTROL
  if ((textPos || "").includes("top")) {
    textZoneControl = `
Top 25% of frame reserved for text,
Face positioned slightly lower,
Clear text-safe zone at top,
`;
  }

  else if (textPos.includes("bottom")) {
    textZoneControl = `
Bottom 25% reserved for bold text,
Face positioned slightly higher,
Clear text-safe zone at bottom,
`;
  }

  else {
    textZoneControl = `
Text positioned beside subject in negative space,
Avoid overlapping face,
`;
  }

} else {
  // fallback
  framing = "Extreme close-up face filling 70% of frame,";
  compositionControl = "Centered subject composition,";
  textZoneControl = "Text placed beside subject,";
}

if (context.dna && context.dna.avg_face_score > 60) {
  framing = "Medium close-up, face filling 50% of frame,";
} else {
  framing = "Waist up framing, environment visible,";
}
  if (context.dna) {
  dnaBoost += `(${context.dna.dominant_lighting_style} lighting:1.3),\n`;
  dnaBoost += `(${context.dna.dominant_composition_style}:1.2),\n`;
  dnaBoost += `(Text at ${context.dna.dominant_text_position}:1.2),\n`;
}

  if (context.dna) {
    if (context.dna.dominant_art_style) {
  dnaBoost += `${context.dna.dominant_art_style} style,\n`;
}

if (context.dna.dominant_render_type) {
  dnaBoost += `${context.dna.dominant_render_type} rendering,\n`;
}

if (context.dna.dominant_distortion) {
  dnaBoost += `${context.dna.dominant_distortion} character proportion,\n`;
}

if (context.dna.dominant_color_palette) {
  dnaBoost += `${context.dna.dominant_color_palette} color palette,\n`;
}


    if (context.dna.avg_color_score > 75) {
      dnaBoost += "Extremely vibrant saturated colors, aggressive contrast,\n";
    }

    if (context.dna.avg_readability_score > 75) {
      dnaBoost += "Huge bold text, maximum readability for mobile,\n";
    }

    if (context.dna.avg_emotion_score > 75) {
      dnaBoost += "Extreme emotional exaggeration, dramatic facial intensity,\n";
    }



    if (context.dna.dominant_text_position) {
      dnaBoost += `Thumbnail text positioned at ${context.dna.dominant_text_position},\n`;
    }

    if (context.dna.avg_realism_level > 60) {
      dnaBoost += "Ultra realistic photo style,\n";
    } else {
      dnaBoost += "Slightly stylized cinematic look,\n";
    }
  }
// 🎭 MOOD ADAPTIVE EXPRESSION
let moodExpression = `(strong emotional expression:${emotionWeight}),`;

if (context.mood?.includes("serious")) {
  moodExpression = `(focused intense stare:${emotionWeight}),`;
}

if (context.mood?.includes("sad")) {
  moodExpression = `(subtle emotional sadness:${emotionWeight}),`;
}

if (context.mood?.includes("educational")) {
  moodExpression = `(confident explanatory expression:${emotionWeight}),`;
}

if (context.mood?.includes("excited")) {
  moodExpression = `(extreme shocked face:${emotionWeight}),`;
}

// 🧬 MERGE OBJECTS (video + channel DNA)
let mergedObjects = [
  ...(context.visual_objects || []),
  ...(context.dna?.niche_profile?.common_objects || [])
];

// ลบตัวซ้ำ
mergedObjects = [...new Set(mergedObjects)];

let visualObjectBlock = "";

if (mergedObjects.length > 0) {
  visualObjectBlock = `
Include specific recognizable visual elements:
${mergedObjects.join(", ")}

Make these elements clearly visible and important in the composition.
`;
}
return `
${nicheBlock}

High CTR ${context.category} YouTube thumbnail,

${videoContextBlock}
${visualObjectBlock}
 ${vectorControl}
${blueprintBlock}
${layoutDNA}

${templateInstruction}

${colorLock}

${backgroundLock}

${microRefinement}

${styleSignatureBlock}

${structuralInjection}

${colorVectorLock}

${geometryControl}

${dnaBoost}

Main subject: ${context.focus},
${framing}
Head near top frame edge,
Foreground dominance,

${moodExpression}
Strong expression clarity,

Thumbnail text: "${context.text_hook}",
Large bold readable text,
2-4 words only,
High contrast yellow or white typography,

Clear foreground subject,
Separated midground,
Strong subject-background separation,

Optimized for small mobile view clarity,
${context.dna?.camera_distance_estimate > 0.6
  ? "Medium lens perspective,"
  : "Wide angle perspective,"}
(slight caricature exaggeration:${exaggerationWeight}),
Foreground subject oversized,
Environment readable and colorful,
`;
}


function buildNegativePrompt() {
  return `
blurry,
low quality,
extra fingers,
bad anatomy,
watermark,
text artifacts,
logo,

neon circle,
neon ring behind head,
perfect circular halo,
symmetrical background pattern,

low contrast,
flat lighting,
boring expression,
small face,
far camera,
realistic documentary style,
`;
}

/* =========================
   🔥 YOUTUBE FETCH
========================= */
async function getYouTubeData(videoId) {
  const response = await axios.get(
    "https://www.googleapis.com/youtube/v3/videos",
    {
      params: {
        part: "snippet,statistics",
        id: videoId,
        key: YOUTUBE_API_KEY,
      },
    }
  );

  if (!response.data.items.length) {
    throw new Error("Video not found");
  }

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
  const ctrWeight = 0.5;
  const hookWeight = 0.3;
  const emotionWeight = 0.2;

  return (
    analysis.ctr_score * ctrWeight +
    analysis.hook_strength * 10 * hookWeight +
    analysis.emotion_intensity * 10 * emotionWeight
  );
}
async function getChannelIdFromVideo(videoId) {
  try {
    const response = await axios.get(
      "https://www.googleapis.com/youtube/v3/videos",
      {
        params: {
          part: "snippet",
          id: videoId,
          key: process.env.YOUTUBE_API_KEY,
        },
      }
    );

    const items = response.data.items;
    if (!items || items.length === 0) {
      throw new Error("Video not found");
    }

    return items[0].snippet.channelId;
  } catch (error) {
    console.error("Error getting channelId:", error.message);
    throw error;
  }
}
function findNodeByType(workflow, type) {
  return Object.keys(workflow).find(
    key => workflow[key].class_type === type
  );
}
/* =========================
   🚀 GENERATE THUMBNAILS
========================= */
app.get("/generate-thumbnail", async (req, res) => {
  try {
    const videoId = req.query.video_id;
    const version = req.query.workflow_version || "thumbmagic_core_v1.json";

    const baseWorkflow = JSON.parse(
      fs.readFileSync(`./workflows/${version}`, "utf-8")
    );

    const youtubeData = await getYouTubeData(videoId);

// 1️⃣ หา channel ก่อน
const channelId = await getChannelIdFromVideo(videoId);

// 2️⃣ โหลด DNA ก่อน
const channelDNA = loadChannelDNA(channelId);

console.log("Channel ID:", channelId);
console.log("Loaded DNA:", channelDNA);

// 3️⃣ ค่อยวิเคราะห์ content พร้อม DNA
const analysis = await analyzeContent(youtubeData, channelDNA);

if (!analysis) {
  return res.json({ error: "Content analysis failed" });
}

console.log("Channel ID:", channelId);
console.log("Loaded DNA:", channelDNA);
// 🔥 Generate Blueprint ก่อน
// 🔥 Generate Blueprint ก่อน
let blueprint = await generateVisualBlueprint(
  {
    topic: youtubeData.title,
    emotion: analysis.mood,
    focus: analysis.focus,
    objects: analysis.visual_objects
  },
  channelDNA
);

if (!blueprint) {
  console.log("⚠️ Blueprint fallback used");
  blueprint = {
    framing: "Extreme close-up",
    subject_type: "face",
    camera_angle: "front",
    emotion_type: analysis.mood || "intense",
    lighting_type: channelDNA?.dominant_lighting_style || "cinematic",
    background_type: "blurred colorful",
    text_zone: channelDNA?.dominant_text_position || "side",
    visual_priority_order: ["face", "text", "object"]
  };
}
    const context = {
  template: analysis.template || "story_cards",
  category: analysis.category || "general",
  focus: analysis.focus || "",
  mood: analysis.mood || "",
  style: analysis.style || "",
  text_hook: analysis.text_hook || "",
  visual_objects: analysis.visual_objects || [],
  dna: channelDNA,
blueprint: blueprint,
  // 🔥 เพิ่มส่วนนี้
  videoTitle: youtubeData.title,
  videoDescription: youtubeData.description,
  videoTags: youtubeData.tags,
  viewCount: youtubeData.viewCount,
  likeCount: youtubeData.likeCount
};

    
    const negativePrompt = buildNegativePrompt();

    
    

    const variations = [];

const textOptions = Array.isArray(analysis.text_variations) && analysis.text_variations.length
  ? analysis.text_variations
  : [analysis.text_hook || "Must Watch"];

for (let i = 0; i < textOptions.length; i++) {

console.log("Generating variation:", textOptions[i]);

  const workflow = JSON.parse(JSON.stringify(baseWorkflow));

  const dynamicContext = {
    ...context,
    text_hook: textOptions[i]
  };

  const finalPrompt = buildPrompt(dynamicContext);

  const kSamplerNode = findNodeByType(workflow, "KSampler");

  let positiveNode = null;
let negativeNode = null;

for (let key of Object.keys(workflow)) {
  if (workflow[key].class_type === "CLIPTextEncode") {
    if (!positiveNode) {
      positiveNode = key;
    } else {
      negativeNode = key;
    }
  }
}

  if (kSamplerNode) {
    workflow[kSamplerNode].inputs.seed = Math.floor(Math.random() * 1000000000);
  }

  if (!positiveNode || !negativeNode) {
  return res.json({ error: "Workflow missing CLIPTextEncode nodes" });
}
console.log("==== FINAL PROMPT ====");
console.log(finalPrompt);
workflow[positiveNode].inputs.text = finalPrompt;
workflow[negativeNode].inputs.text = negativePrompt;
  

  const comfyResponse = await axios.post(
    `${COMFY_URL}/prompt`,
    { prompt: workflow }
  );

  const promptId = comfyResponse.data.prompt_id;

console.log("Prompt ID:", promptId);

  let imageFilename = null;
  let attempts = 0;

  while (!imageFilename && attempts < 45) {
  await new Promise((r) => setTimeout(r, 2000));
  attempts++;

  try {
    const history = await axios.get(`${COMFY_URL}/history/${promptId}`);
    const outputs = history.data[promptId]?.outputs;

    if (outputs) {
      for (let nodeId in outputs) {
        if (outputs[nodeId].images) {
          imageFilename = outputs[nodeId].images[0].filename;
          break;
        }
      }
    }
  } catch (err) {
    console.error("Polling error:", err.message);
    break;
  }
}

if (!imageFilename) {
  console.log("⚠️ Generation timeout for prompt:", promptId);
  continue; // ข้ามไป variation ถัดไป
}

  const imageUrl = `${COMFY_URL}/view?filename=${imageFilename}`;

const imageResponse = await axios.get(imageUrl, {
  responseType: "arraybuffer"
});

const imageBuffer = Buffer.from(imageResponse.data);

let processedImage = imageBuffer;

// ==============================
// 🔥 APPLY CHANNEL DNA COLOR PIPELINE
// ==============================

console.log("🔥 Applying DNA color pipeline...");

// 1️⃣ Histogram Matching ก่อน
if (channelDNA?.avg_color_histogram_256?.rHist?.length === 256) {

  console.log("🔥 Applying REAL histogram matching...");

  processedImage = await matchHistogram(
    processedImage,
    channelDNA.avg_color_histogram_256
  );

} else {
  console.log("⚠️ Histogram missing — skipping histogram match");
}
// 2️⃣ Color Transfer แบบ weight
if (channelDNA?.avg_color_stats?.r) {
  processedImage = await colorTransfer(
    processedImage,
    channelDNA.avg_color_stats,
    0.4
  );
}


// 🔥 วิเคราะห์ด้วยภาพที่ถูกปรับสีแล้ว
const visionAnalysis = await analyzeImageWithVision(processedImage);
if (!visionAnalysis) continue;
  variations.push({
    text: textOptions[i],
    image: imageUrl,
    base_score: calculateFinalScore(analysis),
    vision_score: visionAnalysis.visual_ctr_score,
    face_score: visionAnalysis.face_score,
    emotion_score: visionAnalysis.emotion_score,
    color_score: visionAnalysis.color_score,
    readability_score: visionAnalysis.readability_score
  });

}
  

let winner = null;

if (variations.length > 0) {
  winner = variations.reduce((best, current) =>
  current.vision_score > best.vision_score ? current : best
);
if (winner && channelDNA) {
  updateDNAFromWinner(channelId, winner, channelDNA);
}
}
    res.json({
  status: "completed",
  title: youtubeData.title,
  overall_score: calculateFinalScore(analysis),
  ctr_score: analysis.ctr_score,
  hook_strength: analysis.hook_strength,
  overall_text_score: calculateFinalScore(analysis),
  emotion_intensity: analysis.emotion_intensity,
  winner: winner,
  thumbnails: variations
});

  } catch (err) {
    console.error(err);
    res.json({ error: err.message });
  }
});

app.listen(9000, () => {
  console.log("Server running on http://localhost:9000");
});