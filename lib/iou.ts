// Lightweight identity tracker: IoU matching between frames + EMA smoothing.
//
// The YOLO detector produces boxes per frame. This module stitches them into
// a persistent "target" by keeping only the detection that (a) shares the
// locked class and (b) overlaps the previous box above a threshold. Two
// identical cups in frame? Only one inherits the face.
//
// Smoothing is a straight exponential moving average on (cx, cy, w, h). The
// knob is one alpha: low = glassy but laggy, high = snappy but jittery. The
// brief calls out 0.4 as a sensible start and we honor it.

export type Box = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
};

export function makeBox(cx: number, cy: number, w: number, h: number): Box {
  const hw = w / 2;
  const hh = h / 2;
  return { cx, cy, w, h, x1: cx - hw, y1: cy - hh, x2: cx + hw, y2: cy + hh };
}

export function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  if (x2 <= x1 || y2 <= y1) return 0;
  const inter = (x2 - x1) * (y2 - y1);
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const union = areaA + areaB - inter;
  return union > 0 ? inter / union : 0;
}

// Distance between box centers, normalized by the previous box's diagonal.
// Used as a tie-breaker when two same-class detections both clear the IoU
// gate — prefer the one whose center moved least.
export function centerDistNorm(prev: Box, cand: Box): number {
  const dx = cand.cx - prev.cx;
  const dy = cand.cy - prev.cy;
  const prevDiag = Math.hypot(prev.w, prev.h);
  return prevDiag > 0 ? Math.hypot(dx, dy) / prevDiag : Infinity;
}

export type Candidate = Box & { classId: number; score: number };

// Pick the candidate most likely to be the same object as `prev`. Same class
// required. Ranks by IoU first, then center-distance on ties.
export function matchTarget<C extends Candidate>(
  candidates: readonly C[],
  prev: Box & { classId: number },
  minIou = 0.3,
  // Locked tracks pin identity from the VLM, not YOLO's per-frame class.
  // Pass true to match by geometry alone and let class flips pass through.
  ignoreClass = false
): C | null {
  let best: C | null = null;
  let bestIou = minIou;
  let bestDist = Infinity;
  for (const c of candidates) {
    if (!ignoreClass && c.classId !== prev.classId) continue;
    const ov = iou(c, prev);
    if (ov < minIou) continue;
    const d = centerDistNorm(prev, c);
    // Primary: higher IoU wins by a wide margin. Secondary: closer center.
    if (
      ov > bestIou + 0.05 ||
      (Math.abs(ov - bestIou) <= 0.05 && d < bestDist)
    ) {
      best = c;
      bestIou = ov;
      bestDist = d;
    }
  }
  return best;
}

// Scalar EMA. `alpha` is the weight on the new sample.
// Calling `reset(v)` snaps the state to v (use on reacquisition to avoid a
// visible sliding-in after occlusion).
export class EMAFilter {
  private prev: number | null = null;
  constructor(public alpha: number = 0.4) {}

  update(v: number): number {
    if (this.prev === null) {
      this.prev = v;
    } else {
      this.prev = v * this.alpha + this.prev * (1 - this.alpha);
    }
    return this.prev;
  }

  reset(v?: number): void {
    this.prev = v ?? null;
  }

  get value(): number | null {
    return this.prev;
  }
}

// Bundle of four scalar EMAs tracking a bounding box as (cx, cy, w, h). Using
// center+size (not corners) keeps width/height jitter independent of position
// jitter — important because YOLO's width tends to be noisier than its center.
export type BoxEMA = {
  cx: EMAFilter;
  cy: EMAFilter;
  w: EMAFilter;
  h: EMAFilter;
};

// Split alphas — position is fast (responsive to motion) while size is slow
// (kills the "breathing" effect from bbox edge jitter). Objects rarely change
// size abruptly in real use, so a slow size EMA costs little and buys a lot
// of visual stability.
export function newBoxEMA(posAlpha = 0.4, sizeAlpha = 0.15): BoxEMA {
  return {
    cx: new EMAFilter(posAlpha),
    cy: new EMAFilter(posAlpha),
    w: new EMAFilter(sizeAlpha),
    h: new EMAFilter(sizeAlpha),
  };
}

export function smoothBox(ema: BoxEMA, b: Box): Box {
  return makeBox(
    ema.cx.update(b.cx),
    ema.cy.update(b.cy),
    ema.w.update(b.w),
    ema.h.update(b.h)
  );
}

export function seedBoxEMA(ema: BoxEMA, b: Box): void {
  ema.cx.reset(b.cx);
  ema.cy.reset(b.cy);
  ema.w.reset(b.w);
  ema.h.reset(b.h);
}

// Given a point in the original detection box, express it as offsets relative
// to the box center in normalized-by-box-size units. These are the face
// placement ratios stored at lock time and replayed each frame against the
// newest smoothed box — the math at the heart of Phase 1.5/2.4 in the brief.
export type Anchor = {
  rx: number; // (point.x - box.cx) / box.w, so rx=0 is center, ±0.5 is edge
  ry: number;
};

export function anchorFromPoint(point: { x: number; y: number }, box: Box): Anchor {
  return {
    rx: box.w > 0 ? (point.x - box.cx) / box.w : 0,
    ry: box.h > 0 ? (point.y - box.cy) / box.h : 0,
  };
}

export function applyAnchor(anchor: Anchor, box: Box): { x: number; y: number } {
  return {
    x: box.cx + anchor.rx * box.w,
    y: box.cy + anchor.ry * box.h,
  };
}
