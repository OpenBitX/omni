# Testing v2 end-to-end

v2 is the native phone frontend; the parent Next.js app is the backend. Both
the phone and your Mac must be on the **same Wi-Fi network** so the phone
can reach `http://<your-mac-LAN-IP>:3000` over LAN.

## One-time setup

```sh
# 1. Install native-app deps
cd react-native-executorch
npm install

# 2. Confirm the parent backend is set up
cd ..
cat .env.local            # should have OPENAI / ZHIPU / CEREBRAS / FISH keys

# 3. Find your Mac's LAN IP (remember the number, you'll need it in step 5)
ipconfig getifaddr en0
# → e.g. 192.168.1.42
```

## Start the backend (parent app) bound to LAN

```sh
cd /Users/harryedwards/hackathon
npm run dev -- -H 0.0.0.0 -p 3000
```

`-H 0.0.0.0` binds every interface so the phone (not just localhost) can
connect. You should see `ready - local: http://localhost:3000, network:
http://192.168.1.42:3000`.

Smoke-test from another terminal that the new HTTP wrappers are live:
```sh
curl -s -XPOST http://localhost:3000/api/generate-line \
  -H 'content-type: application/json' \
  -d '{"voiceId":"d75c270eaee14c8aa1e9e980cc37cf1b","description":"a chipped red mug with a faint coffee ring"}' | jq
```
Expect a JSON with `{line, voiceId, description}`. If you see 404, the
parent dev server didn't pick up the new `app/api/*` files — kill and
restart it.

## Point the native app at the backend

```sh
cd /Users/harryedwards/hackathon/react-native-executorch
cp .env.example .env
# edit .env and set EXPO_PUBLIC_API_BASE=http://192.168.1.42:3000
#   ^ use the IP you got from ipconfig, NOT localhost
```

## Build + run the dev client

Expo Go will not work — vision-camera, executorch, and audio-api are native
modules. A one-time `prebuild` generates iOS/Android projects, then every
subsequent launch is `npm run ios` / `npm run android`.

```sh
# Generate native iOS + Android projects
npm run prebuild

# Run on a physical iPhone (USB-connected, trusted):
npm run ios -- --device
# or on Android (USB-debugging, adb-detected):
npm run android -- --device
```

First launch downloads the YOLO26N_SEG model (~9 MB) from SWM's HuggingFace
mirror; subsequent launches use the cached `.pte`. The loading overlay
shows percent complete.

## What "working" looks like

1. **Camera preview** fills the screen; no yellow "red box" error from Metro
2. After ~5–10 s the loading overlay disappears — model ready
3. **Tap an object**: a thinking-dots bubble pops, 1–3 s later a cartoon
   face appears over the object and speaks a line. Voice should match the
   object's vibe (mug → Peter Griffin, laptop → Elon, teddy → Anime Girl…)
4. **Move the object** (or yourself): the face rides it at 60 fps between
   inferences — no visible stutter, no wrong-object hijack
5. **Tap a second object**: second face appears; both can talk over each
   other
6. **Retap an existing face**: ~200 ms later a new line plays in the same
   voice (persona pinned — it's Cerebras text-only, not a vision call)
7. **Hold the "hold to talk" button** and say something: on release, the
   object replies. Transcript + reply should both show in console logs
8. **Tap "clear"**: all faces vanish, mic button greys out

## Latency targets on the same LAN

| Leg                                 | Target     | Notes                          |
|-------------------------------------|------------|--------------------------------|
| On-device YOLO26N_SEG @ 384         | 30–90 ms   | iPhone 12+ / Snapdragon 8xx    |
| `assessObject` (GLM)                | 2–5 s      | Runs in parallel with line     |
| `generateLine` first tap (bundled)  | 1.5–3 s    | GPT-4o-mini vision             |
| `generateLine` retap (Cerebras)     | 150–400 ms | Text-only against persona card |
| TTS first-byte (Fish, LAN)          | 400–900 ms | Buffered then decoded          |
| `converseWithObject` (Cerebras+STT) | 600–1200 ms| Depends on audio length        |

## Troubleshooting

- **"Network request failed"** from the phone → your `EXPO_PUBLIC_API_BASE`
  is `localhost` (simulator-only) or the wrong LAN IP. Double-check with
  `ipconfig getifaddr en0`. Mac firewall can also block port 3000; System
  Settings → Network → Firewall.
- **Model download stalls at <100%** → bad network; relaunch, it resumes.
- **"camera permission denied"** → iOS: Settings → Tracker → Camera;
  Android: App Info → Permissions.
- **Face appears but no voice** → likely TTS env missing. Check backend
  logs for `[tts]` lines; confirm `FISH_API_KEY` in `.env.local`.
- **Sluggish preview** on Android → drop `SEG_INPUT_SIZE` in
  `components/tracker.tsx` from 384 to the closest smaller option the
  `useInstanceSegmentation` hook exposes via `getAvailableInputSizes()`.

## Telemetry

Every tap/converse turn has a `turnId` that threads through the backend
logs. Grep one turn end-to-end:
```sh
# In the backend terminal
grep '#abc123' .  # replace abc123 with the turnId shown in Metro logs
```
