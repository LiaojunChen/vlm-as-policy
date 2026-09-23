"""Probe PhysBrain point grounding on a real RoboTwin RGB observation."""

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image


def first_point(text):
    match = re.search(r'"point_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', text)
    return [float(match.group(1)), float(match.group(2))] if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18765/v1")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    array = np.asarray(image)
    height, width = array.shape[:2]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    masks = {
        "red": (array[:, :, 0] > 150) & (array[:, :, 1] < 110) & (array[:, :, 2] < 110),
        "green": (array[:, :, 1] > 140) & (array[:, :, 0] < 130) & (array[:, :, 2] < 130),
    }
    result = {"image": str(args.image.resolve()), "image_size": [width, height], "queries": []}
    for color, mask in masks.items():
        ys, xs = np.nonzero(mask)
        center = [float(xs.mean()), float(ys.mean())] if len(xs) else None
        prompt = (f"Point to the center of the {color} block. "
                  'The answer should be presented in JSON format as follows: [{"point_2d": [x, y]}].')
        request = {"model": "physbrain1.5-8b-local", "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}], "temperature": 0, "max_tokens": 128}
        started = time.monotonic()
        response = requests.post(args.endpoint + "/chat/completions", json=request, timeout=90)
        response.raise_for_status()
        data = response.json()
        raw = data.get("physbrain_raw_content", data["choices"][0]["message"]["content"])
        point = first_point(raw)
        pixel = [point[0] * width / 1000, point[1] * height / 1000] if point else None
        error = float(np.hypot(pixel[0] - center[0], pixel[1] - center[1])) if pixel and center else None
        result["queries"].append({"color": color, "prompt": prompt, "raw_response": raw,
                                  "point_1000": point, "predicted_pixel": pixel,
                                  "mask_center_pixel": center, "pixel_error": error,
                                  "mask_pixel_count": int(len(xs)),
                                  "elapsed_s": time.monotonic() - started})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
