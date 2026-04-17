type FaceProps = {
  mouth: number; // 0 closed, 1 wide open
  blink: number; // 0 open, 1 shut
  eyeX: number; // -1..1 gaze offset
  eyeY: number;
};

export function Face({ mouth, blink, eyeX, eyeY }: FaceProps) {
  const eyeOpen = Math.max(0.04, 1 - blink);
  const pupilVisible = eyeOpen > 0.3;
  const dx = eyeX * 6;
  const dy = eyeY * 6;
  const m = Math.min(1, Math.max(0, mouth));
  const smileOp = 1 - Math.min(1, m * 4);
  const openOp = Math.min(1, m * 4);
  const openRy = Math.max(0.5, m * 22);

  return (
    <svg
      width="200"
      height="200"
      viewBox="-100 -100 200 200"
      className="drop-shadow-[0_6px_10px_rgba(0,0,0,0.55)]"
      style={{ overflow: "visible" }}
    >
      <ellipse
        cx={-28}
        cy={-18}
        rx={22}
        ry={26 * eyeOpen}
        fill="white"
        stroke="black"
        strokeWidth={3}
      />
      {pupilVisible && (
        <>
          <circle cx={-24 + dx} cy={-12 + dy} r={10} fill="black" />
          <circle cx={-20 + dx} cy={-16 + dy} r={3} fill="white" />
        </>
      )}

      <ellipse
        cx={28}
        cy={-18}
        rx={22}
        ry={26 * eyeOpen}
        fill="white"
        stroke="black"
        strokeWidth={3}
      />
      {pupilVisible && (
        <>
          <circle cx={32 + dx} cy={-12 + dy} r={10} fill="black" />
          <circle cx={36 + dx} cy={-16 + dy} r={3} fill="white" />
        </>
      )}

      <path
        d="M -30 30 Q 0 55 30 30"
        fill="none"
        stroke="black"
        strokeWidth={5}
        strokeLinecap="round"
        opacity={smileOp}
      />
      <ellipse
        cx={0}
        cy={32}
        rx={20 + m * 6}
        ry={openRy}
        fill="black"
        opacity={openOp}
      />
    </svg>
  );
}
