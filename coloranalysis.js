const sharp = require("sharp");

/* =========================================================
   1️⃣ CREATE HISTOGRAM (256 bins per channel)
========================================================= */

async function getHistogram(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .raw()
    .toBuffer({ resolveWithObject: true });

  const bins = 256;
  const rHist = new Array(bins).fill(0);
  const gHist = new Array(bins).fill(0);
  const bHist = new Array(bins).fill(0);

  for (let i = 0; i < data.length; i += info.channels) {
    rHist[data[i]]++;
    gHist[data[i + 1]]++;
    bHist[data[i + 2]]++;
  }

  return { rHist, gHist, bHist };
}

/* =========================================================
   2️⃣ CREATE CDF
========================================================= */

function createCDF(hist) {
  const cdf = [];
  let cumulative = 0;
  const total = hist.reduce((a, b) => a + b, 0);

  for (let i = 0; i < hist.length; i++) {
    cumulative += hist[i];
    cdf.push(cumulative / total);
  }

  return cdf;
}

/* =========================================================
   3️⃣ BUILD LUT FROM SOURCE → TARGET
========================================================= */
function buildLUT(sourceCDF, targetCDF) {
  const lut = new Array(256);

  for (let i = 0; i < 256; i++) {
    let j = 0;
    while (j < 255 && targetCDF[j] < sourceCDF[i]) {
      j++;
    }
    lut[i] = j;
  }

  return lut;
}
function buildCDF(hist) {
  const cdf = [];
  let cumulative = 0;

  for (let i = 0; i < 256; i++) {
    cumulative += hist[i];
    cdf[i] = cumulative;
  }

  return cdf;
}

/* =========================================================
   4️⃣ REAL HISTOGRAM MATCHING
========================================================= */

async function matchHistogram(imageBuffer, targetHist) {
  // targetHist ต้องเป็น:
  // { rHist: [...256], gHist: [...256], bHist: [...256] }

  const sourceHist = await getHistogram(imageBuffer);

  const sourceRCDF = createCDF(sourceHist.rHist);
  const sourceGCDF = createCDF(sourceHist.gHist);
  const sourceBCDF = createCDF(sourceHist.bHist);

  const targetRCDF = createCDF(targetHist.rHist);
  const targetGCDF = createCDF(targetHist.gHist);
  const targetBCDF = createCDF(targetHist.bHist);

  const rLUT = buildLUT(sourceRCDF, targetRCDF);
  const gLUT = buildLUT(sourceGCDF, targetGCDF);
  const bLUT = buildLUT(sourceBCDF, targetBCDF);

  const { data, info } = await sharp(imageBuffer)
    .raw()
    .toBuffer({ resolveWithObject: true });

  for (let i = 0; i < data.length; i += info.channels) {
    data[i]     = rLUT[data[i]];
    data[i + 1] = gLUT[data[i + 1]];
    data[i + 2] = bLUT[data[i + 2]];
  }

  return sharp(data, {
  raw: {
    width: info.width,
    height: info.height,
    channels: info.channels
  }
})
.toFormat("png")
.toBuffer();
}

/* =========================================================
   5️⃣ COLOR TRANSFER (Mean + Std)
========================================================= */

async function getMeanStd(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .raw()
    .toBuffer({ resolveWithObject: true });

  const r = [], g = [], b = [];

  for (let i = 0; i < data.length; i += info.channels) {
    r.push(data[i]);
    g.push(data[i + 1]);
    b.push(data[i + 2]);
  }

  const calc = (arr) => {
    const mean = arr.reduce((a, b) => a + b) / arr.length;
    const std = Math.sqrt(
      arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length
    );
    return { mean, std };
  };

  return { r: calc(r), g: calc(g), b: calc(b) };
}

async function colorTransfer(imageBuffer, targetStats, weight = 0.4) {
  const source = await getMeanStd(imageBuffer);

  const scaleR = 1 + weight * ((targetStats.r.std / source.r.std) - 1);
  const scaleG = 1 + weight * ((targetStats.g.std / source.g.std) - 1);
  const scaleB = 1 + weight * ((targetStats.b.std / source.b.std) - 1);

  const shiftR = weight * (targetStats.r.mean - source.r.mean);
  const shiftG = weight * (targetStats.g.mean - source.g.mean);
  const shiftB = weight * (targetStats.b.mean - source.b.mean);

  return sharp(imageBuffer)
  .linear([scaleR, scaleG, scaleB], [shiftR, shiftG, shiftB])
  .jpeg({ quality: 95 })
  .toBuffer();
}

/* =========================================================
   EXPORT
========================================================= */

module.exports = {
  matchHistogram,
  colorTransfer,
  getHistogram,
  getMeanStd
};