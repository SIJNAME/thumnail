const sharp = require("sharp");

async function getHistogram(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const channels = info.channels;
  const rHist = new Array(256).fill(0);
  const gHist = new Array(256).fill(0);
  const bHist = new Array(256).fill(0);

  for (let i = 0; i < data.length; i += channels) {
    rHist[data[i]] += 1;
    gHist[data[i + 1]] += 1;
    bHist[data[i + 2]] += 1;
  }

  return { rHist, gHist, bHist };
}

function buildCdf(hist) {
  const cdf = new Array(hist.length).fill(0);
  let running = 0;
  const total = hist.reduce((a, b) => a + b, 0) || 1;
  for (let i = 0; i < hist.length; i++) {
    running += hist[i];
    cdf[i] = running / total;
  }
  return cdf;
}

function buildLut(srcHist, targetHist) {
  const srcCdf = buildCdf(srcHist);
  const targetCdf = buildCdf(targetHist);
  const lut = new Array(256).fill(0);

  for (let i = 0; i < 256; i++) {
    let j = 0;
    for (; j < 255 && targetCdf[j] < srcCdf[i]; j += 1) {}
    lut[i] = j;
  }

  return lut;
}

async function matchHistogram(imageBuffer, targetHist) {
  const { data, info } = await sharp(imageBuffer)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const source = await getHistogram(imageBuffer);
  const lutR = buildLut(source.rHist, targetHist.rHist);
  const lutG = buildLut(source.gHist, targetHist.gHist);
  const lutB = buildLut(source.bHist, targetHist.bHist);

  const channels = info.channels;
  const out = Buffer.from(data);

  for (let i = 0; i < out.length; i += channels) {
    out[i] = lutR[out[i]];
    out[i + 1] = lutG[out[i + 1]];
    out[i + 2] = lutB[out[i + 2]];
  }

  return sharp(out, {
    raw: {
      width: info.width,
      height: info.height,
      channels: info.channels,
    },
  })
    .png()
    .toBuffer();
}

module.exports = { getHistogram, matchHistogram };
