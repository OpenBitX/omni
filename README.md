# duidui

**Tap anything through your camera. It grows a face. It talks back.**

Point at a mug, a sofa, a traffic cone — tap it, and a cartoon face locks on
and starts chatting in character. Talk back with the mic and it'll reply.
Works with up to three objects at once.

---

## Get started

You need **Node 20+** and **three API keys** — all important:

1. **[Zhipu GLM](https://open.bigmodel.cn/)** — vision + writing (the brains)
2. **[Fish.audio](https://fish.audio/)** — cloned character voices (the voice)
3. **[OpenAI](https://platform.openai.com/)** — Whisper mic talk-back + TTS fallback

```bash
git clone https://github.com/OpenBitX/duidui.git
cd duidui
npm install
cp .env.example .env.local
```

Open `.env.local` and paste all three:

```ini
ZHIPU_API_KEY=your-zhipu-key
FISH_API_KEY=your-fish-key
FISH_REFERENCE_ID=your-fish-voice-id
OPENAI_API_KEY=your-openai-key
```

Then start the web app:

```bash
npm run dev          # → http://localhost:3000
```

And — only if you want to use `/mirror` — start the Python backend in a
second terminal:

```bash
npm run server       # → http://localhost:8000
```

Open <http://localhost:3000>, let it use your camera, tap something. 🎉

---

## Commands

| What | Command | Port |
| --- | --- | --- |
| Run the web app (Tracker + Mirror UI) | `npm run dev` | `3000` |
| Run the Python backend (only for `/mirror`) | `npm run server` | `8000` |
| Production build | `npm run build` | — |
| Start production server | `npm run start` | `3000` |
| Typecheck | `npm run typecheck` | — |

---

## How it runs

**Everything happens in your browser** except the API calls.

The YOLO object detector, the tracking loop, the face rendering, the mouth
animation — all client-side, no server process required for Tracker. The only
things that leave your machine are three server-action round-trips:

1. **Assess** — GLM-5V looks at the tapped crop and decides where the face goes
2. **Speak** — GLM-4v-flash writes the line, Fish/OpenAI synthesizes the voice
3. **Converse** — your mic audio → Whisper → GLM reply → TTS (when you talk back)

`/mirror` is the exception — it streams frames over WebSocket to the Python
backend on `:8000` for dlib landmark detection + OpenCV compositing.

---

## What each key does

All three keys matter — the app works best with the full set:

| Key | What it powers |
| --- | --- |
| `ZHIPU_API_KEY` | GLM-5V vision (decides where the face goes) + GLM-4v-flash in-character dialogue |
| `FISH_API_KEY` + `FISH_REFERENCE_ID` | Fish.audio `s1` — the primary voice (cloned, character-specific) |
| `OPENAI_API_KEY` | Whisper (mic talk-back transcription) + `tts-1`/`nova` TTS fallback |

Skip Fish → voices fall back to OpenAI TTS. Skip both TTS keys → caption-only
mode. Skip OpenAI → mic talk-back stops working.

---

## Running on your phone

The main use-case is your phone's rear camera. Browsers require HTTPS to
access the camera on a non-localhost address, so the easiest path is a tunnel:

```bash
npm run dev
# in another terminal:
npx ngrok http 3000
```

Open the `https://…ngrok.app` URL on your phone. Done.

First tap downloads the on-device YOLO model (~9.4 MB); cached after that.

---

## The two apps

### 🎯 Tracker — `/` (the main one)

Tap an object, face appears on it. Everything runs in your browser
(onnxruntime-web for detection, GLM for writing, Fish/OpenAI for voice). No
backend process needed.

### 🪞 Mirror — `/mirror` (optional)

Paste your own eyes and mouth onto a base image (an orange, something you
upload, etc.) in real time. This one needs the Python backend running.

```bash
# one-time setup
python3 -m venv server/.venv
server/.venv/bin/pip install \
  opencv-python dlib imutils numpy openai fastapi "uvicorn[standard]" \
  python-multipart websockets python-dotenv

# grab the landmarks model (~100 MB, not in git)
curl -L http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
  -o server/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 server/shape_predictor_68_face_landmarks.dat.bz2

# then, any time you want Mirror:
npm run server
```

On macOS, dlib builds from source — you need `cmake`:

```bash
brew install cmake
```

Start both `npm run dev` and `npm run server`, then visit
<http://localhost:3000/mirror>.

---

## Project layout

```
app/
  page.tsx              Tracker (main app)
  mirror/page.tsx       Mirror
  actions.ts            Server actions — assess, speak, converse
components/
  tracker.tsx           Tracker UI + tracking loop
  mirror.tsx            Mirror client
  face-voice.tsx        Talking-face renderer
lib/
  yolo.ts               Browser YOLO detector
  iou.ts                Identity tracking + anchor math
public/
  models/               YOLO .onnx weights
  ort/                  ORT WASM runtime (auto-populated)
  facevoice/            Eyes video + mouth shape PNGs
server/
  server.py             FastAPI + dlib + OpenCV (Mirror only)
  bases/                Base images + coord cache
```

Deeper architecture notes live in [`CLAUDE.md`](./CLAUDE.md).

---

## Stack

Next.js 15 · React 19 · TypeScript · Tailwind v4 · onnxruntime-web · GLM-5V /
GLM-4v-flash · Fish.audio · OpenAI Whisper + TTS · FastAPI · dlib · OpenCV

---

## Troubleshooting

**Camera won't start on my phone** — you need HTTPS. Use `ngrok` or similar.

**`/ort/*.wasm` 404 in the console** — run `npm install` again (the
`postinstall` script copies ORT files into `public/ort/`).

**"Caption only" mode** — add `FISH_API_KEY` or `OPENAI_API_KEY` for audio.

**Mic button does nothing** — `OPENAI_API_KEY` is required (Whisper).

**Mirror says "server offline"** — run `npm run server` in another terminal.
It'll reconnect automatically.

**dlib install fails** — `brew install cmake` on macOS, then retry.

---

## License

MIT.
