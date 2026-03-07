const axios = require('axios');
const fs = require('fs');
require('dotenv').config();

const COMFY_URL = process.env.COMFY_URL || "http://127.0.0.1:8000";
const YOUTUBE_API_KEY = process.env.YOUTUBE_API_KEY;

async function run() {
  try {
    const videoId = "IAMyWV6Fo6c";
    console.log("Testing YouTube...");
    const yt = await axios.get("https://www.googleapis.com/youtube/v3/videos", {
      params: { part: "snippet,statistics", id: videoId, key: YOUTUBE_API_KEY }
    });
    console.log("YouTube success");

    console.log("Testing ComfyUI workflow prompt...");
    const baseWorkflow = JSON.parse(fs.readFileSync("./workflows/thumbmagic_core_v1.json", "utf-8"));
    const comfyResponse = await axios.post(`${COMFY_URL}/prompt`, {
       prompt: baseWorkflow,
       client_id: "thumbnail-ai"
    });
    console.log("ComfyUI success");

  } catch (err) {
    console.error("ERROR from:", err?.config?.url);
    console.error(err.message);
    if (err.response) {
      console.error(JSON.stringify(err.response.data, null, 2));
    }
  }
}
run();
