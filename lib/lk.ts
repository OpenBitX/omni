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

// Image pyramid for multi-scale tracking. Level 0 is full resolution; each
// level thereafter is half-size via 2x2 box average. Pyramid LK searches
// for motion at coarse levels first (where a 40-pixel displacement looks
// like a 10-pixel one) and refines at finer levels — so point tracking
// survives large inter-frame motion that breaks single-level LK.
export type Pyramid = { levels: Gray[] };

function downsample2x(g: Gray): Gray {
  const sw = g.width;
  const sh = g.height;
  const w = Math.max(1, sw >> 1);
  const h = Math.max(1, sh >> 1);
  const src = g.data;
  const out = new Uint8ClampedArray(w * h);
  for (let y = 0; y < h; y++) {
    const sy = y << 1;
    for (let x = 0; x < w; x++) {
      const sx = x << 1;
      const v =
        (src[sy * sw + sx] +
          src[sy * sw + sx + 1] +
          src[(sy + 1) * sw + sx] +
          src[(sy + 1) * sw + sx + 1]) >>
        2;
      out[y * w + x] = v;
    }
  }
  return { data: out, width: w, height: h };
}

export function buildPyramid(gray: Gray, depth: number): Pyramid {
  const levels: Gray[] = [gray];
  for (let i = 1; i < depth; i++) {
    const p = levels[i - 1];
    if (p.width < 16 || p.height < 16) break;
    levels.push(downsample2x(p));
  }
  return { levels };
}

// Track one point at one pyramid level. Refines `guess` toward the location
// in `curr` that matches the `(anchorX, anchorY)` patch in `prev`. Returns
// the refined position (same coordinate space as inputs), whether it
// converged inside the frame, and the post-fit patch SAD.
function trackPointAtLevel(
  prev: Gray,
  curr: Gray,
  ix: Float32Array,
  iy: Float32Array,
  anchorX: number,
  anchorY: number,
  guessX: number,
  guessY: number,
  winHalf: number,
  maxIters: number,
  tol: number
): { x: number; y: number; found: boolean; sad: number } {
  const w = prev.width;
  const h = prev.height;

  let gxx = 0;
  let gyy = 0;
  let gxy = 0;
  for (let dy = -winHalf; dy <= winHalf; dy++) {
    for (let dx = -winHalf; dx <= winHalf; dx++) {
      const gx = sampleF(ix, w, h, anchorX + dx, anchorY + dy);
      const gy = sampleF(iy, w, h, anchorX + dx, anchorY + dy);
      gxx += gx * gx;
      gyy += gy * gy;
      gxy += gx * gy;
    }
  }
  const det = gxx * gyy - gxy * gxy;
  if (det < 1e-4) return { x: guessX, y: guessY, found: false, sad: 0 };
  const invDet = 1 / det;

  let cx = guessX;
  let cy = guessY;
  let ok = true;
  for (let k = 0; k < maxIters; k++) {
    let bx = 0;
    let by = 0;
    for (let dy = -winHalf; dy <= winHalf; dy++) {
      for (let dx = -winHalf; dx <= winHalf; dx++) {
        const it =
          sample(curr, cx + dx, cy + dy) - sample(prev, anchorX + dx, anchorY + dy);
        const gx = sampleF(ix, w, h, anchorX + dx, anchorY + dy);
        const gy = sampleF(iy, w, h, anchorX + dx, anchorY + dy);
        bx += gx * it;
        by += gy * it;
      }
    }
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
    return { x: cx, y: cy, found: false, sad: 0 };
  }

  let sad = 0;
  for (let dy = -winHalf; dy <= winHalf; dy++) {
    for (let dx = -winHalf; dx <= winHalf; dx++) {
      sad += Math.abs(
        sample(curr, cx + dx, cy + dy) - sample(prev, anchorX + dx, anchorY + dy)
      );
    }
  }
  const avgSad = sad / ((winHalf * 2 + 1) * (winHalf * 2 + 1));
  return { x: cx, y: cy, found: true, sad: avgSad };
}

// Pyramid LK: track `pts` from `prev` to `curr`. Input and output point
// coordinates are in full-resolution (level-0) space. The search runs
// coarse-to-fine, so large displacements converge.
export function trackLK(
  prev: Pyramid,
  curr: Pyramid,
  pts: Pt[],
  opts?: {
    winHalf?: number;
    iters?: number;
    tolerance?: number;
    sadThreshold?: number;
  }
): TrackedPt[] {
  const winHalf = opts?.winHalf ?? 4;
  const maxIters = opts?.iters ?? 12;
  const tol = opts?.tolerance ?? 0.03;
  const sadThreshold = opts?.sadThreshold ?? 40;
  const depth = Math.min(prev.levels.length, curr.levels.length);

  // Precompute prev gradients at every level once per call.
  const grads = prev.levels
    .slice(0, depth)
    .map((g) => computeGradients(g));

  const out: TrackedPt[] = new Array(pts.length);
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const topScale = 1 << (depth - 1);
    let cx = p.x / topScale;
    let cy = p.y / topScale;
    let found = true;
    let finalSad = 0;

    for (let L = depth - 1; L >= 0; L--) {
      const scale = 1 << L;
      const anchorX = p.x / scale;
      const anchorY = p.y / scale;
      const r = trackPointAtLevel(
        prev.levels[L],
        curr.levels[L],
        grads[L].ix,
        grads[L].iy,
        anchorX,
        anchorY,
        cx,
        cy,
        winHalf,
        maxIters,
        tol
      );
      cx = r.x;
      cy = r.y;
      if (!r.found) {
        // Return result in full-resolution coords for caller consistency.
        cx *= scale;
        cy *= scale;
        found = false;
        break;
      }
      if (L === 0) finalSad = r.sad;
      else {
        cx *= 2;
        cy *= 2;
      }
    }

    out[i] = { x: cx, y: cy, found: found && finalSad < sadThreshold };
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

// Returns the similarity equivalent to applying `first`, then `second`:
//   composeTransforms(first, second)(p) = second(first(p))
// A similarity is complex multiplication T(p) = c·p + t with c = a + i·b.
// So c_new = c2·c1 and t_new = c2·t1 + t2.
export function composeTransforms(first: Transform, second: Transform): Transform {
  return {
    a: second.a * first.a - second.b * first.b,
    b: second.a * first.b + second.b * first.a,
    tx: second.a * first.tx - second.b * first.ty + second.tx,
    ty: second.b * first.tx + second.a * first.ty + second.ty,
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
