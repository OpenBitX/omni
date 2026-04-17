"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  imageSrc: string;
  topText: string;
  bottomText: string;
  onReady?: (dataUrl: string) => void;
  className?: string;
};

export default function MemeCanvas({
  imageSrc,
  topText,
  bottomText,
  onReady,
  className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [dataUrl, setDataUrl] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      if (cancelled) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      // Force 9:16 output regardless of source ratio
      const W = 1080;
      const H = 1920;
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, W, H);
      // Cover-fit the source image into 9:16
      const srcW = img.naturalWidth || W;
      const srcH = img.naturalHeight || H;
      const srcRatio = srcW / srcH;
      const dstRatio = W / H;
      let dw = W,
        dh = H,
        dx = 0,
        dy = 0;
      if (srcRatio > dstRatio) {
        dh = H;
        dw = H * srcRatio;
        dx = (W - dw) / 2;
      } else {
        dw = W;
        dh = W / srcRatio;
        dy = (H - dh) / 2;
      }
      ctx.drawImage(img, dx, dy, dw, dh);
      drawMemeText(ctx, topText, "top", W, H);
      drawMemeText(ctx, bottomText, "bottom", W, H);
      const url = canvas.toDataURL("image/png");
      setDataUrl(url);
      onReady?.(url);
    };
    img.onerror = () => {};
    img.src = imageSrc;
    return () => {
      cancelled = true;
    };
  }, [imageSrc, topText, bottomText, onReady]);

  return (
    <>
      <canvas ref={canvasRef} className="hidden" />
      {dataUrl && (
        <img
          src={dataUrl}
          alt={`${topText} ${bottomText}`.trim()}
          className={className}
        />
      )}
    </>
  );
}

function drawMemeText(
  ctx: CanvasRenderingContext2D,
  text: string,
  position: "top" | "bottom",
  width: number,
  height: number
) {
  if (!text) return;

  const paddingX = Math.round(width * 0.05);
  const maxWidth = width - paddingX * 2;

  // Size relative to height; start large, shrink to fit
  let fontSize = Math.round(height * 0.085);
  const minFont = Math.round(height * 0.045);
  const fontFamily =
    '"Impact", "Anton", "Oswald", "Arial Black", "Helvetica Inserat", sans-serif';

  const wrap = (size: number): string[] => {
    ctx.font = `900 ${size}px ${fontFamily}`;
    const words = text.split(/\s+/);
    const lines: string[] = [];
    let current = "";
    for (const word of words) {
      const candidate = current ? `${current} ${word}` : word;
      if (ctx.measureText(candidate).width <= maxWidth) {
        current = candidate;
      } else {
        if (current) lines.push(current);
        current = word;
      }
    }
    if (current) lines.push(current);
    return lines;
  };

  let lines: string[] = [];
  while (fontSize >= minFont) {
    lines = wrap(fontSize);
    const widestFits = lines.every(
      (l) => ctx.measureText(l).width <= maxWidth
    );
    const maxLines = 4;
    if (widestFits && lines.length <= maxLines) break;
    fontSize -= 2;
  }

  ctx.font = `900 ${fontSize}px ${fontFamily}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.lineJoin = "round";
  ctx.miterLimit = 2;

  const lineHeight = Math.round(fontSize * 1.05);
  const strokeWidth = Math.max(4, Math.round(fontSize * 0.12));

  const x = width / 2;
  const blockHeight = lineHeight * lines.length;
  const topY = Math.round(height * 0.04) + fontSize;
  const bottomY = height - Math.round(height * 0.04);

  const startY =
    position === "top"
      ? topY
      : bottomY - blockHeight + lineHeight;

  for (let i = 0; i < lines.length; i++) {
    const y = startY + i * lineHeight;
    // Slight shadow for readability over busy backgrounds
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.45)";
    ctx.shadowBlur = Math.round(fontSize * 0.18);
    ctx.shadowOffsetY = Math.round(fontSize * 0.04);
    ctx.strokeStyle = "#000";
    ctx.lineWidth = strokeWidth;
    ctx.strokeText(lines[i], x, y);
    ctx.restore();

    ctx.fillStyle = "#fff";
    ctx.fillText(lines[i], x, y);
  }
}
