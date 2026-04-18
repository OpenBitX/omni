"""Ask GPT-4o where eyes and a mouth should sit on a 512x512 version of the image.

Usage:
    OPENAI_API_KEY=... .venv/bin/python detect_coords.py <image_path>

Writes coords.json next to main.py with this shape:
    {"left_eye": [x, y], "right_eye": [x, y], "mouth": [x, y]}
Coordinates are CENTER points in the 512x512 canvas that main.py uses.
"""

import base64
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

if len(sys.argv) != 2:
    print("usage: detect_coords.py <image>", file=sys.stderr)
    sys.exit(1)

img_path = Path(sys.argv[1])
img_bytes = img_path.read_bytes()
b64 = base64.b64encode(img_bytes).decode()
ext = img_path.suffix.lstrip(".").lower() or "jpeg"
if ext == "jpg":
    ext = "jpeg"

prompt = (
    "This image will be resized to a 512x512 square and used as the base for a "
    "face composite. Return JSON giving pixel coordinates (at the 512x512 scale) "
    "for where to paste a LEFT eye, a RIGHT eye, and a MOUTH so the result reads "
    "as a face. Coordinates are the CENTER of each feature.\n"
    "Schema (return ONLY this JSON, nothing else):\n"
    '{"left_eye": [x, y], "right_eye": [x, y], "mouth": [x, y]}\n'
    "Pick positions that make sense for the subject in the image — e.g. on a "
    "face or object, place them where a face would naturally appear. Eyes should "
    "roughly be on the same y, separated horizontally. Mouth below the eyes."
)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
resp = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{ext};base64,{b64}",
                        "detail": "low",
                    },
                },
            ],
        }
    ],
)

raw = resp.choices[0].message.content
coords = json.loads(raw)

out = Path(__file__).parent / "coords.json"
out.write_text(json.dumps(coords, indent=2))
print(json.dumps(coords, indent=2))
print(f"\nwrote {out}", file=sys.stderr)
