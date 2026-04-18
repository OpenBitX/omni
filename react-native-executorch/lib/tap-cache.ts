import type { Detection } from "@/lib/detector";
import { EXCLUDED_CLASS_IDS } from "@/lib/detector";

// Pick the smallest-area detection that contains the tap. Smaller boxes are
// usually the correct semantic target when two are nested (cup inside the
// region of a dining table). Falls back to nearest-center if nothing
// contains the tap. Ported verbatim from the browser tracker.
export function pickTappedDetection(
  dets: readonly Detection[],
  tapX: number,
  tapY: number
): Detection | null {
  let best: Detection | null = null;
  let bestArea = Infinity;
  for (const d of dets) {
    if (EXCLUDED_CLASS_IDS.has(d.classId)) continue;
    if (tapX < d.x1 || tapX > d.x2 || tapY < d.y1 || tapY > d.y2) continue;
    const area = (d.x2 - d.x1) * (d.y2 - d.y1);
    if (area < bestArea) {
      bestArea = area;
      best = d;
    }
  }
  if (best) return best;
  let nearest: Detection | null = null;
  let nearestD = Infinity;
  for (const d of dets) {
    if (EXCLUDED_CLASS_IDS.has(d.classId)) continue;
    const dx = d.cx - tapX;
    const dy = d.cy - tapY;
    const dist = Math.hypot(dx, dy);
    if (dist < nearestD) {
      nearestD = dist;
      nearest = d;
    }
  }
  return nearest;
}

// Convert a source-frame rectangle (detection's x1..y2 in native frame
// pixels) into preview-view coordinates. The camera preview uses `cover`
// fit, so there's letterboxing on one axis we need to account for, just
// like sourceBoxToElement in the browser tracker.
export type ViewBox = { left: number; top: number; width: number; height: number };

export function sourceBoxToView(
  box: { x1: number; y1: number; x2: number; y2: number },
  frame: { width: number; height: number },
  view: { width: number; height: number }
): ViewBox {
  const vAspect = view.width / Math.max(1, view.height);
  const fAspect = frame.width / Math.max(1, frame.height);
  let dispW: number;
  let dispH: number;
  let offX: number;
  let offY: number;
  if (fAspect > vAspect) {
    dispH = frame.height;
    dispW = frame.height * vAspect;
    offX = (frame.width - dispW) / 2;
    offY = 0;
  } else {
    dispW = frame.width;
    dispH = frame.width / vAspect;
    offX = 0;
    offY = (frame.height - dispH) / 2;
  }
  const sx = view.width / dispW;
  const sy = view.height / dispH;
  return {
    left: (box.x1 - offX) * sx,
    top: (box.y1 - offY) * sy,
    width: (box.x2 - box.x1) * sx,
    height: (box.y2 - box.y1) * sy,
  };
}

export function viewPointToSource(
  x: number,
  y: number,
  frame: { width: number; height: number },
  view: { width: number; height: number }
): { x: number; y: number } {
  const vAspect = view.width / Math.max(1, view.height);
  const fAspect = frame.width / Math.max(1, frame.height);
  let dispW: number;
  let dispH: number;
  let offX: number;
  let offY: number;
  if (fAspect > vAspect) {
    dispH = frame.height;
    dispW = frame.height * vAspect;
    offX = (frame.width - dispW) / 2;
    offY = 0;
  } else {
    dispW = frame.width;
    dispH = frame.width / vAspect;
    offX = 0;
    offY = (frame.height - dispH) / 2;
  }
  return {
    x: offX + (x / view.width) * dispW,
    y: offY + (y / view.height) * dispH,
  };
}

export function sourcePointToView(
  p: { x: number; y: number },
  frame: { width: number; height: number },
  view: { width: number; height: number }
): { x: number; y: number } {
  const vAspect = view.width / Math.max(1, view.height);
  const fAspect = frame.width / Math.max(1, frame.height);
  let dispW: number;
  let dispH: number;
  let offX: number;
  let offY: number;
  if (fAspect > vAspect) {
    dispH = frame.height;
    dispW = frame.height * vAspect;
    offX = (frame.width - dispW) / 2;
    offY = 0;
  } else {
    dispW = frame.width;
    dispH = frame.width / vAspect;
    offX = 0;
    offY = (frame.height - dispH) / 2;
  }
  const sx = view.width / dispW;
  const sy = view.height / dispH;
  return { x: (p.x - offX) * sx, y: (p.y - offY) * sy };
}
