import axios from "axios";
import { env } from "../config/env";
import { withRetry } from "../utils/retry";
import { ExternalServiceError } from "../utils/errors";

export async function getVideoMeta(videoId: string) {
  return withRetry(async () => {
    try {
      const response = await axios.get("https://www.googleapis.com/youtube/v3/videos", {
        params: {
          part: "snippet,statistics",
          id: videoId,
          key: env.YOUTUBE_API_KEY,
        },
      });
      return response.data;
    } catch (err) {
      throw new ExternalServiceError("YouTube API failed", "youtube");
    }
  });
}
