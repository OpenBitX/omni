"""Annoying-orange-style face composite with auto-detected feature placement.

Usage:
    .venv/bin/python app.py                      # uses orange.jpg
    .venv/bin/python app.py path/to/image.png    # uses any image

On first run with a given image, asks GPT-4o for eye/mouth positions and
caches them in <image>.coords.json. Delete that file to re-detect.

Needs OPENAI_API_KEY in the environment.

Press q to quit.
"""

import base64
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import dlib
import numpy as np
from imutils import face_utils, resize

CANVAS = 512


def ask_gpt_for_coords(img_path: Path) -> dict:
    from openai import OpenAI

    img_bytes = img_path.read_bytes()
    b64 = base64.b64encode(img_bytes).decode()
    ext = img_path.suffix.lstrip(".").lower() or "jpeg"
    if ext == "jpg":
        ext = "jpeg"

    prompt = (
        f"This image will be resized to {CANVAS}x{CANVAS} and used as the base "
        "for a face composite. Return JSON giving pixel coordinates (at the "
        f"{CANVAS}x{CANVAS} scale) for where to paste a LEFT eye, a RIGHT eye, "
        "and a MOUTH so the result reads as a face. Coordinates are the CENTER "
        "of each feature.\n"
        "Schema (return ONLY this JSON):\n"
        '{"left_eye": [x, y], "right_eye": [x, y], "mouth": [x, y]}\n'
        "Pick positions that make sense for the subject. Eyes roughly on the "
        "same y, separated horizontally. Mouth below the eyes."
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
    return json.loads(resp.choices[0].message.content)


def load_or_detect_coords(img_path: Path) -> dict:
    img_hash = hashlib.md5(img_path.read_bytes()).hexdigest()[:8]
    cache = img_path.with_suffix(f".{img_hash}.coords.json")
    if cache.exists():
        print(f"using cached coords: {cache.name}")
        return json.loads(cache.read_text())
    print(f"asking GPT-4o for feature coords on {img_path.name}...")
    coords = ask_gpt_for_coords(img_path)
    cache.write_text(json.dumps(coords, indent=2))
    print(f"got {coords}, cached to {cache.name}")
    return coords


def main():
    img_path = Path(sys.argv[1] if len(sys.argv) > 1 else "orange.jpg")
    if not img_path.exists():
        sys.exit(f"image not found: {img_path}")

    base = cv2.imread(str(img_path))
    if base is None:
        sys.exit(f"cv2 could not read: {img_path}")
    base = cv2.resize(base, (CANVAS, CANVAS))

    coords = load_or_detect_coords(img_path)
    le_target = tuple(coords["left_eye"])
    re_target = tuple(coords["right_eye"])
    mo_target = tuple(coords["mouth"])

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        faces = detector(img)
        result = base.copy()

        if len(faces) > 0:
            face = faces[0]
            shape = face_utils.shape_to_np(predictor(img, face))

            le_x1, le_y1 = shape[36, 0], shape[37, 1]
            le_x2, le_y2 = shape[39, 0], shape[41, 1]
            le_margin = int((le_x2 - le_x1) * 0.18)

            re_x1, re_y1 = shape[42, 0], shape[43, 1]
            re_x2, re_y2 = shape[45, 0], shape[47, 1]
            re_margin = int((re_x2 - re_x1) * 0.18)

            left_eye_img = img[le_y1 - le_margin:le_y2 + le_margin,
                               le_x1 - le_margin:le_x2 + le_margin].copy()
            right_eye_img = img[re_y1 - re_margin:re_y2 + re_margin,
                                re_x1 - re_margin:re_x2 + re_margin].copy()

            if left_eye_img.size and right_eye_img.size:
                left_eye_img = resize(left_eye_img, width=160)
                right_eye_img = resize(right_eye_img, width=160)

                result = cv2.seamlessClone(
                    left_eye_img, result,
                    np.full(left_eye_img.shape[:2], 255, left_eye_img.dtype),
                    le_target, cv2.NORMAL_CLONE,
                )
                result = cv2.seamlessClone(
                    right_eye_img, result,
                    np.full(right_eye_img.shape[:2], 255, right_eye_img.dtype),
                    re_target, cv2.NORMAL_CLONE,
                )

            mouth_x1, mouth_y1 = shape[48, 0], shape[50, 1]
            mouth_x2, mouth_y2 = shape[54, 0], shape[57, 1]
            mouth_margin = int((mouth_x2 - mouth_x1) * 0.1)

            mouth_img = img[mouth_y1 - mouth_margin:mouth_y2 + mouth_margin,
                            mouth_x1 - mouth_margin:mouth_x2 + mouth_margin].copy()

            if mouth_img.size:
                mouth_img = resize(mouth_img, width=320)
                result = cv2.seamlessClone(
                    mouth_img, result,
                    np.full(mouth_img.shape[:2], 255, mouth_img.dtype),
                    mo_target, cv2.NORMAL_CLONE,
                )

        cv2.imshow("result", result)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
