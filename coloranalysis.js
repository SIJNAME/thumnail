const sharp = require("sharp");

const BINS = 256;
const MAX_ANALYSIS_WIDTH = 256;

function clamp(value, min = 0, max = 255) {
  return Math.max(min, Math.min(max, value));
}

function normalize01(value, min = 0, max = 1) {
  if (!Number.isFinite(value)) return 0;
  return clamp((value - min) / Math.max(1e-12, max - min), 0, 1);
}

function smoothHistogram(hist, windowSize = 5) {
  const half = Math.floor(windowSize / 2);
  const out = new Array(hist.length).fill(0);

  for (let i = 0; i < hist.length; i += 1) {
    let sum = 0;
    let count = 0;

    for (let k = i - half; k <= i + half; k += 1) {
      const idx = clamp(k, 0, hist.length - 1);
      sum += hist[idx];
      count += 1;
    }

    out[i] = sum / Math.max(1, count);
  }

  return out;
}

function createCDF(hist) {
  const total = hist.reduce((a, b) => a + b, 0);
  const safeTotal = Math.max(1e-12, total);
  const cdf = new Array(hist.length).fill(0);

  let cumulative = 0;
  for (let i = 0; i < hist.length; i += 1) {
    cumulative += hist[i];
    cdf[i] = cumulative / safeTotal;
  }
  return cdf;
}

function buildLUT(sourceCDF, targetCDF) {
  const lut = new Array(BINS).fill(0);

  for (let i = 0; i < BINS; i += 1) {
    let j = 0;
    while (j < BINS - 1 && targetCDF[j] < sourceCDF[i]) j += 1;
    lut[i] = j;
  }

  return lut;
}

async function toAnalysisRaw(imageBuffer) {
  const metadata = await sharp(imageBuffer).metadata();
  const width = metadata.width || MAX_ANALYSIS_WIDTH;
  const needResize = width > MAX_ANALYSIS_WIDTH;

  const pipeline = sharp(imageBuffer)
    .rotate()
    .removeAlpha();

  if (needResize) {
    pipeline.resize({ width: MAX_ANALYSIS_WIDTH, withoutEnlargement: true, fit: "inside" });
  }

  return pipeline.raw().toBuffer({ resolveWithObject: true });
}

function computeStatsFromRaw(rawData) {
  const { data, info } = rawData;
  const channels = info.channels;

  const histogram = {
    rHist: new Array(BINS).fill(0),
    gHist: new Array(BINS).fill(0),
    bHist: new Array(BINS).fill(0),
  };

  // Welford accumulators for memory-safe streaming stats
  let pixelCount = 0;
  let meanR = 0;
  let meanG = 0;
  let meanB = 0;
  let m2R = 0;
  let m2G = 0;
  let m2B = 0;

  let meanBrightness = 0;
  let m2Brightness = 0;
  let saturationSum = 0;

  for (let i = 0; i < data.length; i += channels) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];

    histogram.rHist[r] += 1;
    histogram.gHist[g] += 1;
    histogram.bHist[b] += 1;

    pixelCount += 1;

    // RGB channel mean/std (streaming)
    const dR = r - meanR;
    meanR += dR / pixelCount;
    m2R += dR * (r - meanR);

    const dG = g - meanG;
    meanG += dG / pixelCount;
    m2G += dG * (g - meanG);

    const dB = b - meanB;
    meanB += dB / pixelCount;
    m2B += dB * (b - meanB);

    // Brightness and saturation (normalized 0..1 for ML usage)
    const rN = r / 255;
    const gN = g / 255;
    const bN = b / 255;

    const maxC = Math.max(rN, gN, bN);
    const minC = Math.min(rN, gN, bN);
    const brightness = (rN + gN + bN) / 3;
    const saturation = maxC === 0 ? 0 : (maxC - minC) / maxC;

    const dBr = brightness - meanBrightness;
    meanBrightness += dBr / pixelCount;
    m2Brightness += dBr * (brightness - meanBrightness);

    saturationSum += saturation;
  }

  const safeCount = Math.max(pixelCount, 1);
  const variance = (m2) => (safeCount > 1 ? m2 / safeCount : 0);

  return {
    histogram,
    stats: {
      r: { mean: meanR, std: Math.sqrt(Math.max(variance(m2R), 0)) },
      g: { mean: meanG, std: Math.sqrt(Math.max(variance(m2G), 0)) },
      b: { mean: meanB, std: Math.sqrt(Math.max(variance(m2B), 0)) },
    },
    features: {
      brightness: normalize01(meanBrightness),
      saturation: normalize01(saturationSum / safeCount),
      contrast: normalize01(Math.sqrt(Math.max(variance(m2Brightness), 0))),
    },
  };
}

function colorDistance(sourceStats, targetStats) {
  const dR = (sourceStats.r.mean - targetStats.r.mean) / 255;
  const dG = (sourceStats.g.mean - targetStats.g.mean) / 255;
  const dB = (sourceStats.b.mean - targetStats.b.mean) / 255;
  return normalize01(Math.sqrt((dR * dR + dG * dG + dB * dB) / 3));
}

function adaptiveWeight(baseWeight, distance) {
  const base = Number.isFinite(baseWeight) ? baseWeight : 0.4;
  const clampedBase = clamp(base, 0, 1);
  // adapt stronger when color distance is large
  return clamp(0.2 + (clampedBase * 0.5) + (distance * 0.6), 0.15, 0.95);
}

async function getHistogram(imageBuffer) {
  const raw = await toAnalysisRaw(imageBuffer);
  const { histogram } = computeStatsFromRaw(raw);
  return histogram;
}

async function analyzeColorFeatures(imageBuffer, targetStats = null) {
  const raw = await toAnalysisRaw(imageBuffer);
  const computed = computeStatsFromRaw(raw);

  const distance = targetStats ? colorDistance(computed.stats, targetStats) : 0;

  return {
    brightness: computed.features.brightness,
    saturation: computed.features.saturation,
    contrast: computed.features.contrast,
    color_distance_to_target: normalize01(distance),
    stats: computed.stats,
    histogram: computed.histogram,
  };
}

async function matchHistogram(imageBuffer, targetHist) {
  const sourceHist = await getHistogram(imageBuffer);

  const sourceRCDF = createCDF(smoothHistogram(sourceHist.rHist));
  const sourceGCDF = createCDF(smoothHistogram(sourceHist.gHist));
  const sourceBCDF = createCDF(smoothHistogram(sourceHist.bHist));

  const targetRCDF = createCDF(smoothHistogram(targetHist.rHist || new Array(BINS).fill(0)));
  const targetGCDF = createCDF(smoothHistogram(targetHist.gHist || new Array(BINS).fill(0)));
  const targetBCDF = createCDF(smoothHistogram(targetHist.bHist || new Array(BINS).fill(0)));

  const rLUT = buildLUT(sourceRCDF, targetRCDF);
  const gLUT = buildLUT(sourceGCDF, targetGCDF);
  const bLUT = buildLUT(sourceBCDF, targetBCDF);

  const { data, info } = await sharp(imageBuffer)
    .rotate()
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  for (let i = 0; i < data.length; i += info.channels) {
    data[i] = clamp(rLUT[data[i]], 0, 255);
    data[i + 1] = clamp(gLUT[data[i + 1]], 0, 255);
    data[i + 2] = clamp(bLUT[data[i + 2]], 0, 255);
  }

  return sharp(data, {
    raw: {
      width: info.width,
      height: info.height,
      channels: info.channels,
    },
  })
    .png()
    .toBuffer();
}

async function colorTransfer(imageBuffer, targetStats, weight = 0.4) {
  const sourceAnalysis = await analyzeColorFeatures(imageBuffer, targetStats);
  const source = sourceAnalysis.stats;
  const distance = sourceAnalysis.color_distance_to_target;
  const appliedWeight = adaptiveWeight(weight, distance);

  const safe = (v) => (Number.isFinite(v) && v > 1e-6 ? v : 1);

  const scaleR = 1 + appliedWeight * ((safe(targetStats?.r?.std) / safe(source.r.std)) - 1);
  const scaleG = 1 + appliedWeight * ((safe(targetStats?.g?.std) / safe(source.g.std)) - 1);
  const scaleB = 1 + appliedWeight * ((safe(targetStats?.b?.std) / safe(source.b.std)) - 1);

  const shiftR = appliedWeight * ((targetStats?.r?.mean ?? source.r.mean) - source.r.mean);
  const shiftG = appliedWeight * ((targetStats?.g?.mean ?? source.g.mean) - source.g.mean);
  const shiftB = appliedWeight * ((targetStats?.b?.mean ?? source.b.mean) - source.b.mean);

  const { data, info } = await sharp(imageBuffer)
    .rotate()
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  for (let i = 0; i < data.length; i += info.channels) {
    data[i] = clamp(Math.round(data[i] * scaleR + shiftR), 0, 255);
    data[i + 1] = clamp(Math.round(data[i + 1] * scaleG + shiftG), 0, 255);
    data[i + 2] = clamp(Math.round(data[i + 2] * scaleB + shiftB), 0, 255);
  }

  return sharp(data, {
    raw: {
      width: info.width,
      height: info.height,
      channels: info.channels,
    },
  })
    .jpeg({ quality: 95 })
    .toBuffer();
}

module.exports = {
  matchHistogram,
  colorTransfer,
  analyzeColorFeatures,
  getHistogram,
};
