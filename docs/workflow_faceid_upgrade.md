# ComfyUI Workflow Upgrade: SDXL + FaceID + FaceSwap + FaceRestore

ไฟล์ workflow ที่อัปเดตแล้ว:
- `workflows/thumbmagic_core_v1.json`
- `workflow.json`

## 1) New nodes added

### Step 2 — IP-Adapter FaceID identity conditioning
- `LoadImage` (node `12`) โหลด `creator_face.jpg`
- `InsightFaceLoader` (node `13`) สำหรับ face embedding extraction
- `CLIPVisionLoader` (node `14`) สำหรับ image feature encoder
- `IPAdapterUnifiedLoaderFaceID` (node `15`) โหลด FaceID adapter model
- `IPAdapterFaceID` (node `16`) inject identity เข้า SDXL model ก่อนเข้า `KSampler`

### Step 3 — ReActor Face Swap
- `ReActorFaceSwap` (node `20`) สลับหน้า creator ลงบนภาพที่ generate แล้ว
  - `source_image`: `creator_face.jpg`
  - `input_image`: output จาก `VAEDecode` (thumbnail base)

### Step 4 — Face Restore
- `CodeFormerFaceRestore` (node `21`) กู้รายละเอียดใบหน้าหลัง face swap
  - `codeformer_weight`: `0.5`

### Final post-processing
- `ImageColorAdjust` (node `22`) เพิ่ม contrast / saturation / brightness เล็กน้อย
- `ImageScaleBy` (node `23`) upscale `1.2x`
- `SaveImage` (node `9`) บันทึกเป็น prefix `final_thumbnail`

## 2) Graph flow explanation

```text
CheckpointLoaderSimple
  -> LoraLoader -> LoraLoader
  -> CLIPTextEncode (positive/negative)

LoadImage(creator_face.jpg)
  -> InsightFaceLoader
  -> CLIPVisionLoader
  -> IPAdapterUnifiedLoaderFaceID
  -> IPAdapterFaceID (modifies SDXL model with creator identity)

EmptyLatentImage + conditioned model + text conditioning
  -> KSampler
  -> VAEDecode                      (thumbnail_base)
  -> ReActorFaceSwap               (thumbnail_faceswapped)
  -> CodeFormerFaceRestore         (thumbnail_restored)
  -> ImageColorAdjust
  -> ImageScaleBy
  -> SaveImage                     (final_thumbnail.png series)
```

## 3) Required model files

ใส่ไฟล์โมเดลต่อไปนี้ในโฟลเดอร์ที่ extension นั้น ๆ ใช้งาน:

- SDXL checkpoint:
  - `juggernautXL_ragnarokBy.safetensors`
- LoRA:
  - `xieshisanshitu.safetensors`
  - `1708694017811508015.safetensors`
- IP-Adapter FaceID:
  - `ip-adapter-faceid-plusv2_sdxl.bin`
- CLIP Vision model:
  - `ViT-H-14-laion2B-s32B-b79K.safetensors`
- InsightFace / ReActor:
  - `inswapper_128.onnx`
  - RetinaFace detector weights (ตาม ReActor/InsightFace setup)
- Face restore:
  - CodeFormer weights (เช่น `codeformer.pth`)

## 4) Backend usage note

- workflow นี้ใช้ `LoadImage` สำหรับ `creator_face.jpg`
- ถ้าระบบ backend inject controlnet mask อัตโนมัติ ให้แยก node mask loader คนละตัว (ไม่ overwrite node creator face)
