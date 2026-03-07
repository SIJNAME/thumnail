export interface HistogramData {
  rHist: number[];
  gHist: number[];
  bHist: number[];
}

export interface VisionBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface VisionAnalysis {
  visual_ctr_score: number;
  face_score: number;
  emotion_score: number;
  color_score: number;
  readability_score: number;
  subject_position_x: number;
  negative_space_ratio: number;
  face_bbox?: VisionBBox;
  text_bbox?: VisionBBox;
}

export interface ChannelDNA {
  channel_id: string;
  sample_size: number;
  subject_position_x: number;
  negative_space_ratio: number;
  emotion_intensity_avg: number;
  contrast_strength: number;
  avg_color_histogram_256?: HistogramData;
  avg_color_stats?: {
    r: { mean: number; std: number };
    g: { mean: number; std: number };
    b: { mean: number; std: number };
  };
  [key: string]: unknown;
}

export interface ComfyJob {
  jobId: string;
  status: "waiting" | "active" | "completed" | "failed";
  result?: unknown;
  error?: string;
}
