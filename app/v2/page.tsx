// V2 — SAM2-tiny backend-driven tracker. Entirely parallel to the main
// app at `/`; the two share no runtime state, only server actions + TTS
// route. Run the backend with `npm run server:v2` (port 8001).

import { TrackerV2 } from "@/components/tracker-v2";

export const maxDuration = 60;

export default function V2Page() {
  return <TrackerV2 />;
}
