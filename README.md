# duidui

**Tap anything through your camera. It grows a face. It talks back.**

Point your phone at a mug, a sofa, a traffic cone. Tap it. The app detects the
object on-device, locks a cartoon face onto it, writes a line in character, and
speaks it out loud. Talk back with the mic button — it'll reply. Objects stay
tracked as the camera moves; up to three can be alive at once, chattering over
each other.

Two apps live here:

- **Tracker** (`/`) — the main app. Browser-only tracking (YOLO26n-seg via
  onnxruntime-web), GLM-5V for vision + writing, Fish.audio / OpenAI for TTS.
  No backend required beyond Next server actions.
- **Mirror** (`/mirror`) — the original demo. Streams webcam frames to a local
  Python server that uses dlib + OpenCV `seamlessClone` to paste your face
  onto a base image (an orange, your upload, etc.). Optional.

---

## Quick start (Tracker only)

```bash
git clone https://github.com/OpenBitX/duidui.git
cd duidui
npm install
cp .env.example .env.local        # fill in keys — see below
npm run dev
```

Open <http://localhost:3000>, grant camera access, tap something.

### Minimum keys

Edit `.env.local`:

```ini
ZHIPU_API_KEY=...        # required — GLM vision + writing
OPENAI_API_KEY=...       # optional — Whisper (for talk-back) + TTS fallback
FISH_API_KEY=...         # optional — primary TTS (better voice)
FISH_REFERENCE_ID=...    # optional — voice to clone
```

| Key | What breaks without it |
| --- | --- |
| `ZHIPU_API_KEY` | Everything. Required. |
| `OPENAI_API_KEY` | Mic button (Whisper). TTS still works if Fish is set. |
| `FISH_API_KEY` + `FISH_REFERENCE_ID` | Falls back to OpenAI `tts-1` (`nova`). |
| Neither TTS key | Caption-only mode — face still animates, no audio. |

GLM endpoint is auto-picked from key shape (`<hex>.<secret>` → bigmodel.cn,
`sk-…` → api.z.ai). Override via `ZHIPU_BASE_URL` if needed.

---

## Running on your phone

Tracker needs the rear camera and HTTPS (or `localhost`). On the same Wi-Fi:

```bash
npm run dev -- -H 0.0.0.0
```

Then open `https://<your-mac's-ip>:3000` on your phone. Most browsers will
refuse the camera over plain HTTP from a LAN IP — use a tunnel
(`ngrok http 3000`, `cloudflared`, etc.) or Chrome's "Insecure origins treated
as secure" flag for quick testing.

First tap downloads the YOLO model (~9.4 MB). WebGPU is tried first, WASM is
the fallback. On mobile CPU you'll see ~3–8 fps inference; the face
extrapolates between frames so it stays smooth.

---

## Scripts

```bash
npm run dev          # dev server on :3000
npm run build        # production build
npm run start        # run the production build
npm run typecheck    # tsc --noEmit — the only automated gate
npm run server       # FastAPI + WebSocket server for /mirror (see below)
```

`npm install` also runs `scripts/setup-ort.mjs` (postinstall), which copies
onnxruntime-web's WASM runtime into `public/ort/` so the browser can load it
same-origin. Re-run manually if the browser 404s on `/ort/*.wasm`.

There's no lint and no test suite. Typecheck is the gate.

---

## Mirror (optional Python backend)

Only needed if you want `/mirror`.

```bash
# from repo root
python3 -m venv server/.venv
server/.venv/bin/pip install \
  opencv-python dlib imutils numpy openai fastapi "uvicorn[standard]" \
  python-multipart websockets python-dotenv

# dlib landmarks model (~100 MB, gitignored)
curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
  -o server/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 server/shape_predictor_68_face_landmarks.dat.bz2

npm run server       # starts :8000
```

On macOS, dlib builds from source and needs `cmake` via Homebrew:

```bash
brew install cmake
```

Mirror uses the same `ZHIPU_API_KEY` (for base-image coordinate detection) and
reads `.env.local` via python-dotenv. The Next app talks to the Python server
over WebSocket; start both (`npm run dev` + `npm run server`) and visit
<http://localhost:3000/mirror>.

---

## Stack

- **Next.js 15** (App Router), **React 19**, **TypeScript**, **Tailwind v4**
- **onnxruntime-web** for browser-side YOLO26n-seg (WebGPU → WASM fallback)
- **GLM-5V-Turbo** (deep, reasoning) for tap assessment
- **GLM-4v-flash** (fast) for in-character lines + voice replies
- **Fish.audio `s1`** → **OpenAI `tts-1`** TTS ladder
- **OpenAI Whisper** for talk-back transcription
- Python side: **FastAPI**, **dlib**, **OpenCV** (Mirror only)

---

## Project layout

```
app/
  page.tsx              Tracker (main)
  mirror/page.tsx       Mirror
  landing/page.tsx      Static marketing page
  actions.ts            Server actions — assess, speak, converse
components/
  tracker.tsx           Tracker UI + tracking loop
  mirror.tsx            Mirror WS client
  face-voice.tsx        Talking face (eyes video + mouth PNGs)
lib/
  yolo.ts               onnxruntime-web detector
  iou.ts                Identity tracker + anchor math
public/
  models/yolo26n-seg.onnx
  ort/                  ORT WASM runtime (populated by postinstall)
  facevoice/            eyes.webm, shape-A.png … shape-X.png
server/
  server.py             FastAPI + dlib + seamlessClone (Mirror)
  bases/                Base images + coord cache
scripts/
  setup-ort.mjs         Postinstall — copies ORT runtime
  test-*.mjs            Ad-hoc probes (not part of any test suite)
```

See [`CLAUDE.md`](./CLAUDE.md) for the deep architecture notes — tracking loop
internals, prompt design, reseed triggers, the two-transform keyframe scheme,
etc.

---

## Troubleshooting

- **Camera doesn't start on phone** — needs HTTPS on non-localhost. Use a
  tunnel or install a dev cert.
- **`/ort/*.wasm` 404** — run `node scripts/setup-ort.mjs` (or `npm install`).
- **Tracker says "caption only"** — no TTS keys configured. Add `FISH_API_KEY`
  or `OPENAI_API_KEY`.
- **Mic button does nothing** — `OPENAI_API_KEY` is required for Whisper.
- **Mirror shows "server offline"** — `npm run server` in another terminal.
  The chip strip polls `/bases` every 5s and self-heals on reconnect.
- **dlib install fails** — `brew install cmake` then retry the pip install.
- **First tap is slow** — YOLO model downloads (~9.4 MB). Cached after that.

---

## License

MIT.
