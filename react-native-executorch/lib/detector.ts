// Native detection helpers built on react-native-executorch's
// useInstanceSegmentation (YOLO26N_SEG). Preserves the browser lib/yolo.ts
// contract: same Detection shape, same COCO class set, same mask-centroid
// computed downstream so lib/iou.ts anchor math keeps working unchanged.

export const COCO_CLASSES: readonly string[] = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
  "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
  "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
  "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
  "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
  "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
  "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
  "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush",
];

export const PERSON_CLASS_ID = 0;
export const EXCLUDED_CLASS_IDS = new Set<number>([PERSON_CLASS_ID]);

const LABEL_INDEX: Record<string, number> = Object.fromEntries(
  COCO_CLASSES.map((c, i) => [c, i])
);

export function classNameToId(label: string): number {
  return LABEL_INDEX[label] ?? -1;
}

export type Detection = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
  score: number;
  classId: number;
  className: string;
  // Mask centroid is in source-pixel coordinates. Preferred over bbox center
  // as the anchor origin — materially more stable on asymmetric objects
  // (the mug-handle problem).
  maskCentroid?: { x: number; y: number };
  // Pixel count inside the binary mask (proto resolution). Sudden drop →
  // occlusion signal.
  maskArea?: number;
  // Binary silhouette, cropped to the detection's bbox in mask-pixel space.
  // Bytes are 0 (outside) or 255 (inside). Row-major, stride = w.
  mask?: { data: Uint8Array; w: number; h: number };
};

// Raw shape returned by useInstanceSegmentation.forward / runOnFrame.
export type RawSegInstance = {
  bbox: { x1: number; y1: number; x2: number; y2: number };
  label: string;
  score: number;
  mask: Uint8Array; // binary 0/1, full-image resolution when
                    // returnMaskAtOriginalResolution:true, else low-res grid
  maskWidth: number;
  maskHeight: number;
};

// Convert a native instance-seg result into our Detection. srcW/srcH are the
// original frame dimensions (bbox coords already live in that space per the
// RNE docs). Mask centroid is computed once here and cached on the detection
// so the per-frame loop never has to re-scan the mask.
export function normalizeDetection(
  raw: RawSegInstance,
  srcW: number,
  srcH: number
): Detection | null {
  const classId = classNameToId(raw.label);
  if (classId < 0) return null;
  const { x1, y1, x2, y2 } = raw.bbox;
  if (!(x2 > x1) || !(y2 > y1)) return null;
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  const w = x2 - x1;
  const h = y2 - y1;

  // Compute mask centroid in source-pixel space. The mask's coord space is
  // maskWidth × maskHeight covering the full frame (not bbox-cropped). We
  // crop to bbox while scanning so the centroid reflects the object only.
  let cxPix = cx;
  let cyPix = cy;
  let maskArea = 0;
  let croppedMask: { data: Uint8Array; w: number; h: number } | undefined;
  if (raw.mask && raw.maskWidth > 0 && raw.maskHeight > 0) {
    const mw = raw.maskWidth;
    const mh = raw.maskHeight;
    const sx = mw / srcW;
    const sy = mh / srcH;
    const mx1 = Math.max(0, Math.floor(x1 * sx));
    const my1 = Math.max(0, Math.floor(y1 * sy));
    const mx2 = Math.min(mw, Math.ceil(x2 * sx));
    const my2 = Math.min(mh, Math.ceil(y2 * sy));
    const cw = Math.max(1, mx2 - mx1);
    const ch = Math.max(1, my2 - my1);
    const cropped = new Uint8Array(cw * ch);
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    for (let my = my1; my < my2; my++) {
      const srcRow = my * mw;
      const dstRow = (my - my1) * cw;
      for (let mx = mx1; mx < mx2; mx++) {
        const v = raw.mask[srcRow + mx];
        if (v) {
          cropped[dstRow + (mx - mx1)] = 255;
          sumX += mx;
          sumY += my;
          count++;
        }
      }
    }
    if (count > 0) {
      cxPix = (sumX / count) / sx;
      cyPix = (sumY / count) / sy;
      maskArea = count;
      croppedMask = { data: cropped, w: cw, h: ch };
    }
  }

  const det: Detection = {
    x1, y1, x2, y2, cx, cy, w, h,
    score: raw.score,
    classId,
    className: COCO_CLASSES[classId] ?? "?",
  };
  if (croppedMask) {
    det.maskCentroid = { x: cxPix, y: cyPix };
    det.maskArea = maskArea;
    det.mask = croppedMask;
  }
  return det;
}
