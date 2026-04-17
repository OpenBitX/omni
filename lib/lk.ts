// Lucas-Kanade point tracker + Shi-Tomasi corners + similarity fit.
// Single-precision, single-level. Inputs are grayscale Uint8ClampedArray
// at roughly 320px wide — fine without a pyramid because inter-frame
// motion stays small at that size.

export type Gray = {
  data: Uint8ClampedArray;
  width: number;
  height: number;
};

export type Pt = { x: number; y: number };
export type TrackedPt = { x: number; y: number; found: boolean };
export type Corner = { x: number; y: number; score: number };

// Similarity: [ a  -b  tx ]
//             [ b   a  ty ]
export type Transform = { a: number; b: number; tx: number; ty: number };

export function toGray(img: ImageData): Gray {
  const { data, width, height } = img;
  const out = new Uint8ClampedArray(width * height);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    out[j] = (data[i] * 299 + data[i + 1] * 587 + data[i + 2] * 114) / 1000;
  }
  return { data: out, width, height };
}

function sample(g: Gray, x: number, y: number): number {
  const w = g.width;
  const h = g.height;
  if (x < 0) x = 0;
  else if (x > w - 1) x = w - 1;
  if (y < 0) y = 0;
  else if (y > h - 1) y = h - 1;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1 < w ? x0 + 1 : x0;
  const y1 = y0 + 1 < h ? y0 + 1 : y0;
  const fx = x - x0;
  const fy = y - y0;
  const d = g.data;
  const v00 = d[y0 * w + x0];
  const v10 = d[y0 * w + x1];
  const v01 = d[y1 * w + x0];
  const v11 = d[y1 * w + x1];
  const top = v00 + (v10 - v00) * fx;
  const bot = v01 + (v11 - v01) * fx;
  return top + (bot - top) * fy;
}

function sampleF(buf: Float32Array, w: number, h: number, x: number, y: number): number {
  if (x < 0) x = 0;
  else if (x > w - 1) x = w - 1;
  if (y < 0) y = 0;
  else if (y > h - 1) y = h - 1;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1 < w ? x0 + 1 : x0;
  const y1 = y0 + 1 < h ? y0 + 1 : y0;
  const fx = x - x0;
  const fy = y - y0;
  const v00 = buf[y0 * w + x0];
  const v10 = buf[y0 * w + x1];
  const v01 = buf[y1 * w + x0];
  const v11 = buf[y1 * w + x1];
  const top = v00 + (v10 - v00) * fx;
  const bot = v01 + (v11 - v01) * fx;
  return top + (bot - top) * fy;
}

function computeGradients(g: Gray): { ix: Float32Array; iy: Float32Array } {
  const { data, width, height } = g;
  const ix = new Float32Array(width * height);
  const iy = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const xm = x > 0 ? x - 1 : x;
      const xp = x < width - 1 ? x + 1 : x;
      const ym = y > 0 ? y - 1 : y;
      const yp = y < height - 1 ? y + 1 : y;
      ix[y * width + x] = (data[y * width + xp] - data[y * width + xm]) * 0.5;
      iy[y * width + x] = (data[yp * width + x] - data[ym * width + x]) * 0.5;
    }
  }
  return { ix, iy };
}

export function detectCorners(
  gray: Gray,
  opts: {
    maxCorners: number;
    minDistance: number;
    roi?: { x: number; y: number; w: number; h: number };
    threshold?: number;
  }
): Corner[] {
  const { width, height } = gray;
  const { maxCorners, minDistance } = opts;
  const threshold = opts.threshold ?? 8;
  const win = 2; // 5x5 structure window
  const pad = win + 1; // leave room for central-difference gradient

  const x0 = Math.max(pad, Math.floor(opts.roi?.x ?? pad));
  const y0 = Math.max(pad, Math.floor(opts.roi?.y ?? pad));
  const x1 = Math.min(
    width - 1 - pad,
    Math.floor(opts.roi ? opts.roi.x + opts.roi.w : width - 1 - pad)
  );
  const y1 = Math.min(
    height - 1 - pad,
    Math.floor(opts.roi ? opts.roi.y + opts.roi.h : height - 1 - pad)
  );
  if (x1 <= x0 || y1 <= y0) return [];

  const { ix, iy } = computeGradients(gray);
  const scores: Corner[] = [];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      let gxx = 0;
      let gyy = 0;
      let gxy = 0;
      for (let dy = -win; dy <= win; dy++) {
        for (let dx = -win; dx <= win; dx++) {
          const idx = (y + dy) * width + (x + dx);
          const gx = ix[idx];
          const gy = iy[idx];
          gxx += gx * gx;
          gyy += gy * gy;
          gxy += gx * gy;
        }
      }
      const tr = gxx + gyy;
      const det = gxx * gyy - gxy * gxy;
      const disc = Math.sqrt(Math.max(0, tr * tr - 4 * det));
      const minEig = (tr - disc) * 0.5;
      if (minEig > threshold) scores.push({ x, y, score: minEig });
    }
  }

  scores.sort((a, b) => b.score - a.score);

  const out: Corner[] = [];
  const md2 = minDistance * minDistance;
  for (const c of scores) {
    let ok = true;
    for (const k of out) {
      const dx = c.x - k.x;
      const dy = c.y - k.y;
      if (dx * dx + dy * dy < md2) {
        ok = false;
        break;
      }
    }
    if (ok) {
      out.push(c);
      if (out.length >= maxCorners) break;
    }
  }
  return out;
}

export function trackLK(
  prev: Gray,
  curr: Gray,
  pts: Pt[],
  opts?: { winHalf?: number; iters?: number; tolerance?: number }
): TrackedPt[] {
  const winHalf = opts?.winHalf ?? 4;
  const maxIters = opts?.iters ?? 12;
  const tol = opts?.tolerance ?? 0.03;
  const { ix, iy } = computeGradients(prev);
  const w = prev.width;
  const h = prev.height;
  const out: TrackedPt[] = new Array(pts.length);

  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const px = p.x;
    const py = p.y;

    // Structure tensor at the prev location (constant across iterations).
    let gxx = 0;
    let gyy = 0;
    let gxy = 0;
    for (let dy = -winHalf; dy <= winHalf; dy++) {
      for (let dx = -winHalf; dx <= winHalf; dx++) {
        const gx = sampleF(ix, w, h, px + dx, py + dy);
        const gy = sampleF(iy, w, h, px + dx, py + dy);
        gxx += gx * gx;
        gyy += gy * gy;
        gxy += gx * gy;
      }
    }
    const det = gxx * gyy - gxy * gxy;
    if (det < 1e-4) {
      out[i] = { x: px, y: py, found: false };
      continue;
    }
    const invDet = 1 / det;

    let cx = px;
    let cy = py;
    let ok = true;
    for (let k = 0; k < maxIters; k++) {
      let bx = 0;
      let by = 0;
      for (let dy = -winHalf; dy <= winHalf; dy++) {
        for (let dx = -winHalf; dx <= winHalf; dx++) {
          const it = sample(curr, cx + dx, cy + dy) - sample(prev, px + dx, py + dy);
          const gx = sampleF(ix, w, h, px + dx, py + dy);
          const gy = sampleF(iy, w, h, px + dx, py + dy);
          bx += gx * it;
          by += gy * it;
        }
      }
      // δv = -G^-1 b, then v += δv.
      const ux = -invDet * (gyy * bx - gxy * by);
      const uy = -invDet * (-gxy * bx + gxx * by);
      cx += ux;
      cy += uy;
      if (!Number.isFinite(cx) || !Number.isFinite(cy)) {
        ok = false;
        break;
      }
      if (Math.abs(ux) + Math.abs(uy) < tol) break;
    }

    if (!ok || cx < 1 || cy < 1 || cx > w - 2 || cy > h - 2) {
      out[i] = { x: cx, y: cy, found: false };
      continue;
    }

    // Reject if appearance drift is huge — catches occlusions.
    let sad = 0;
    for (let dy = -winHalf; dy <= winHalf; dy++) {
      for (let dx = -winHalf; dx <= winHalf; dx++) {
        sad += Math.abs(sample(curr, cx + dx, cy + dy) - sample(prev, px + dx, py + dy));
      }
    }
    const avgSad = sad / ((winHalf * 2 + 1) * (winHalf * 2 + 1));
    out[i] = { x: cx, y: cy, found: avgSad < 40 };
  }
  return out;
}

export function estimateSimilarity(src: Pt[], dst: Pt[]): Transform | null {
  const n = src.length;
  if (n < 2 || n !== dst.length) return null;

  let sx = 0;
  let sy = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    sx += src[i].x;
    sy += src[i].y;
    dx += dst[i].x;
    dy += dst[i].y;
  }
  sx /= n;
  sy /= n;
  dx /= n;
  dy /= n;

  let num1 = 0;
  let num2 = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i].x - sx;
    const y = src[i].y - sy;
    const xp = dst[i].x - dx;
    const yp = dst[i].y - dy;
    num1 += x * xp + y * yp;
    num2 += x * yp - y * xp;
    den += x * x + y * y;
  }
  if (den < 1e-6) return null;

  const a = num1 / den;
  const b = num2 / den;
  const tx = dx - a * sx + b * sy;
  const ty = dy - b * sx - a * sy;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return { a, b, tx, ty };
}

export function filterOutliers(
  src: Pt[],
  dst: Pt[],
  t: Transform,
  threshold = 3
): { src: Pt[]; dst: Pt[] } {
  const outSrc: Pt[] = [];
  const outDst: Pt[] = [];
  const th2 = threshold * threshold;
  for (let i = 0; i < src.length; i++) {
    const pred = applyTransform(t, src[i]);
    const rx = pred.x - dst[i].x;
    const ry = pred.y - dst[i].y;
    if (rx * rx + ry * ry <= th2) {
      outSrc.push(src[i]);
      outDst.push(dst[i]);
    }
  }
  return { src: outSrc, dst: outDst };
}

export function applyTransform(t: Transform, p: Pt): Pt {
  return {
    x: t.a * p.x - t.b * p.y + t.tx,
    y: t.b * p.x + t.a * p.y + t.ty,
  };
}

export function invertTransform(t: Transform): Transform {
  const det = t.a * t.a + t.b * t.b;
  if (det < 1e-10) return { a: 1, b: 0, tx: 0, ty: 0 };
  const ia = t.a / det;
  const ib = -t.b / det;
  return {
    a: ia,
    b: ib,
    tx: -ia * t.tx + ib * t.ty,
    ty: -ib * t.tx - ia * t.ty,
  };
}
