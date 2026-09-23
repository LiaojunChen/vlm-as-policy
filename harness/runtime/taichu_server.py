"""Minimal OpenAI-compatible endpoint for local ZDTaichu evaluation."""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from json_prefix import JsonPrefix

sys.path.insert(0,str(Path(__file__).resolve().parent/'grammar_vendor'))
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn

MODEL_ROOT = Path(__file__).resolve().parents[2] / "ZDTaichu5.0-9B"
sys.path.insert(0, str(MODEL_ROOT))
from infer import load_model


def load():
    print("Loading ZDTaichu5.0-9B", flush=True)
    processor, model = load_model(str(MODEL_ROOT / "model"))
    processor.tokenizer.apply_chat_template = functools.partial(
        processor.tokenizer.apply_chat_template, enable_thinking=False
    )
    print("Model ready; enable_thinking=False", flush=True)
    return processor, model


PROCESSOR, MODEL = load()
TOKENIZER_DATA=build_token_enforcer_tokenizer_data(PROCESSOR.tokenizer)
LOCK = threading.Lock()


def complete(body: dict) -> dict:
    if body.get("stream"):
        raise ValueError("Streaming is not supported")
    messages = []
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            normalized = []
            for item in content:
                if item.get("type") == "image_url":
                    normalized.append({"type": "image", "image": item["image_url"]["url"]})
                elif item.get("type") == "text":
                    normalized.append(item)
                else:
                    raise ValueError("Only text and image_url content are supported")
            content = normalized
        messages.append({**message, "content": content})
    if not messages:
        raise ValueError("messages is required")

    started = time.monotonic()
    with LOCK, torch.inference_mode():
        inputs = PROCESSOR.from_messages(
            messages, return_tensors="pt", add_vision_id=False
        ).to(MODEL.device)
        prompt_tokens = inputs["input_ids"].shape[1]
        limit = int(body.get("max_tokens", 550))
        temperature = float(body.get("temperature", 0))
        kwargs = {"max_new_tokens": limit, "do_sample": temperature > 0}
        response_format=body.get('response_format',{})
        schema=response_format.get('json_schema',{}).get('schema') if response_format.get('type')=='json_schema' else None
        constraint=None
        if schema is not None:
            if temperature!=0:raise ValueError('JSON-constrained endpoint currently requires greedy temperature=0')
            constraint=build_transformers_prefix_allowed_tokens_fn(TOKENIZER_DATA,JsonSchemaParser(schema))
            kwargs['prefix_allowed_tokens_fn']=constraint
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=0.95, top_k=20)
        output = MODEL.generate(**inputs, **kwargs)
        generated = output[:, prompt_tokens:]
        answer = PROCESSOR.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if schema is not None and JsonPrefix(schema).status(answer)!='complete':
            raise ValueError('Generation budget ended before completing schema-constrained JSON')
        eos_ids = MODEL.generation_config.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        finish_reason = "stop" if int(generated[0, -1]) in eos_ids else "length"
    completion_tokens = generated.shape[1]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "zdtaichu",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
        "x_elapsed_seconds": round(time.monotonic() - started, 3),
        "x_enable_thinking": False,
        "x_json_schema_enforced":schema is not None,
        "x_constraint_backend":"lm-format-enforcer-0.11.3" if constraint else None,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("HTTP " + fmt % args, flush=True)

    def send_json(self, status: HTTPStatus, data: dict):
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(HTTPStatus.OK, {"status": "ok", "model": "zdtaichu",
                "enable_thinking": False, "device": str(MODEL.device),
                "json_schema_enforced":True,
                "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
        else:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 50 * 1024 * 1024:
                raise ValueError("invalid request size")
            body = json.loads(self.rfile.read(size))
            self.send_json(HTTPStatus.OK, complete(body))
        except Exception as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18050)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
