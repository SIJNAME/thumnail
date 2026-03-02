# ระบบ Autonomous Thumbnail Evolution Engine (Node.js)

เอกสารนี้อธิบายระบบฝั่ง Backend ที่พัฒนาต่อจากโค้ดเดิม ให้เป็นระบบสร้าง Thumbnail แบบเรียนรู้ได้เอง (self-learning) ด้วย Channel DNA ซึ่งครอบคลุมด้านเลย์เอาต์ อารมณ์ สี ตัวอักษร และการปรับตัวจากผลลัพธ์จริง

## ฟีเจอร์อัปเกรดหลักที่ทำไว้แล้ว

1. **Spatial Intelligence Layer (แผนที่พื้นที่ภาพ)**
   - เก็บ `subject_density_map` ขนาด 36x64
   - เก็บ `negative_space_map` ขนาด 36x64
   - สร้างจากการสะสมตำแหน่ง `face_bbox` และ `text_bbox` ระหว่างการวิเคราะห์ช่อง (`/analyze-channel`)

2. **Text Zone Enforcement (บังคับโซนข้อความ)**
   - Prompt จะอิง `dominant_text_position`, `text_bbox_avg` และกฎ safe margin จากใบหน้า
   - ถ้าภาพที่ generate แล้วมี text ทับ face เกิน 15% จะข้าม variation นั้น

3. **Face Emotion Conditioning (คุมระดับอารมณ์ใบหน้า)**
   - ใช้ค่า `avg_emotion_score` และ `emotion_intensity_avg` จาก DNA เพื่อกำหนดสไตล์อารมณ์ใน prompt

4. **Typography DNA System (DNA ด้านตัวอักษร)**
   - เก็บค่ารูปแบบตัวอักษรที่เรียนรู้จากช่อง เช่น:
     - `uppercase_ratio`
     - `dominant_text_outline`
     - `dominant_font_weight`
     - `text_area_ratio`
     - `avg_word_count`

5. **Composition Locking (ล็อกองค์ประกอบภาพตามช่อง)**
   - Prompt จะอ้างอิง:
     - `subject_position_x`
     - `camera_distance_estimate`
     - `horizon_line_estimate`
     - `dominant_composition_style`

6. **Prompt DNA Merge Engine**
   - มีฟังก์ชัน `mergeDNA(context, channelDNA)` สำหรับประกอบ baseline แบบมีโครงสร้าง
   - รวมข้อมูลหลักของช่อง เช่น niche, lighting, composition, color palette, art/render style และ realism

7. **Adaptive Reinforcement Learning (เรียนรู้จากผู้ชนะแบบถ่วงน้ำหนัก)**
   - `updateDNAFromWinner()` ใช้สูตร:
     - `new = old + alpha*(winner-old)` โดย `alpha = 0.15`
   - ปรับค่าหลักต่อเนื่อง เช่น:
     - `subject_position_x`
     - `emotion_intensity_avg`
     - `color_saturation_level`
     - `contrast_strength`
     - `negative_space_ratio`
   - จำกัด rolling sample ไม่เกิน 200 ตัวอย่าง

## Extra Requirements ที่รองรับแล้ว

- มี in-memory cache อายุ 24 ชั่วโมงสำหรับ `/analyze-channel`
- มี error handling สำหรับ YouTube quota
- จำกัดงาน generate ผ่าน ComfyUI แบบพร้อมกันสูงสุด 2 งาน
- ตรวจโครงสร้าง DNA schema ก่อนบันทึกไฟล์
- เปิด debug log prompt ได้ด้วย `DEBUG_PROMPT=1`

## API Endpoints

### `GET /analyze-channel?channel_id=<CHANNEL_ID>&controlnet_mask=0|1`
ใช้สำหรับวิเคราะห์ช่อง YouTube และสร้าง/บันทึก Channel DNA

- รับ thumbnail ของช่องมาวิเคราะห์
- สร้างสถิติรวม + spatial map
- บันทึกไฟล์ JSON ในโฟลเดอร์ `channel_dna/`
- ถ้าเปิด `controlnet_mask=1` จะสร้าง mask สำหรับ workflow ที่รองรับ ControlNet

### `GET /generate-thumbnail?video_id=<VIDEO_ID>&workflow_version=thumbmagic_core_v1.json&controlnet_mask=0|1`
ใช้สร้างหลาย thumbnail variation ผ่าน ComfyUI โดยอิง Channel DNA

- สร้าง prompt จาก context + DNA
- ตรวจและบังคับกฎ text/face overlap
- ให้คะแนน variation และเลือกภาพที่ดีที่สุด
- อัปเดต DNA จากผู้ชนะ (winner feedback)

### `GET /system-flow`
ใช้ดูภาพรวม flow ของระบบแบบ closed-loop พร้อมรายการโมดูลหลักที่ backend ใช้งานจริง

## ตัวอย่าง Channel DNA Schema

```json
{
  "channel_id": "UCxxxx",
  "sample_size": 73,
  "niche_profile": {
    "niche": "gaming"
  },
  "face_bbox_avg": { "x": 0.23, "y": 0.19, "w": 0.34, "h": 0.52 },
  "text_bbox_avg": { "x": 0.68, "y": 0.12, "w": 0.26, "h": 0.18 },
  "subject_density_map": [[0.0]],
  "negative_space_map": [[1.0]],
  "subject_position_x": 0.42,
  "camera_distance_estimate": 0.58,
  "horizon_line_estimate": 0.47,
  "avg_emotion_score": 78,
  "emotion_intensity_avg": 74.2,
  "uppercase_ratio": 0.82,
  "dominant_text_outline": 1,
  "dominant_font_weight": "bold",
  "text_area_ratio": 0.12,
  "avg_color_histogram_256": {
    "rHist": [0.0],
    "gHist": [0.0],
    "bHist": [0.0]
  }
}
```

## การตั้งค่า ENV

```env
OPENAI_API_KEY=...
YOUTUBE_API_KEY=...
COMFY_URL=http://127.0.0.1:8188
DEBUG_PROMPT=0
```

## วิธีใช้งาน

```bash
npm install
npm start
```

## ตรวจ syntax

```bash
node --check server.js
node --check coloranalysis.js
```

## รูปแบบ `workflow.json` ที่แนะนำ

ให้วางไฟล์ที่ `workflows/thumbmagic_core_v1.json` (มีตัวอย่างใน repo แล้ว) และควรมีเงื่อนไขหลักดังนี้:

- มี `KSampler` 1 ตัว
- มี `CLIPTextEncode` อย่างน้อย 2 ตัว (positive + negative)
- positive/negative ควรใช้ `clip` chain เดียวกัน (แนะนำให้ชี้ไปที่ LoRA chain เดียวกัน)
- มี `SaveImage` เพื่อให้ ComfyUI ส่งชื่อไฟล์กลับมาได้

หมายเหตุ:
- ระบบ backend จะเติม prompt จริงโดยอัตโนมัติ (เขียนทับ `inputs.text` ของ CLIP node)
- ฝั่ง server รองรับการหา positive/negative node จาก `class_type` และมีการเขียนทับ prompt ตอน runtime
