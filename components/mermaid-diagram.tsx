"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  chart: string;
  className?: string;
};

export default function MermaidDiagram({ chart, className }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [svg, setSvg] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const mermaid = (await import("mermaid")).default;
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        theme: "base",
        fontFamily: "var(--font-geist-sans), ui-sans-serif, system-ui",
        themeVariables: {
          primaryColor: "#ffe4f2",
          primaryTextColor: "#2a1540",
          primaryBorderColor: "#ec4899",
          lineColor: "#c026d3",
          secondaryColor: "#e4d1ff",
          tertiaryColor: "#cfe4ff",
          fontSize: "14px",
        },
      });
      const id = `m-${Math.random().toString(36).slice(2)}`;
      try {
        const { svg } = await mermaid.render(id, chart);
        if (!cancelled) setSvg(svg);
      } catch (err) {
        if (!cancelled) {
          setSvg(
            `<pre style="color:#a00;padding:12px;">mermaid error: ${String(
              err,
            )}</pre>`,
          );
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [chart]);

  return (
    <div
      ref={ref}
      className={className}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
