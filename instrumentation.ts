export function register() {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;
  const g = globalThis as { __tsLogsPatched?: boolean };
  if (g.__tsLogsPatched) return;
  g.__tsLogsPatched = true;

  const stamp = () => {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    const ms = String(d.getMilliseconds()).padStart(3, "0");
    return `${hh}:${mm}:${ss}.${ms}`;
  };

  for (const level of ["log", "warn", "error", "info", "debug"] as const) {
    const orig = console[level].bind(console);
    console[level] = (...args: unknown[]) => orig(`[${stamp()}]`, ...args);
  }
}
