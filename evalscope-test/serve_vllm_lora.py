#!/usr/bin/env python3
"""
Minimal vLLM server with optional LoRA adapters.

Features
- Loads a base Hugging Face model via vLLM.
- Per-request optional LoRA via adapter path (LoRA fine-tune on top of base).
- If no LoRA is provided, serves the original base model.
- Simple HTTP API using FastAPI.

Endpoints
- POST /generate: { prompt, ...sampling, lora_path?, adapter_name? }

Examples
- Base model:
  curl -s -X POST http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"Hello, world!","max_tokens":64}'

- With LoRA (adapter on disk):
  curl -s -X POST http://127.0.0.1:8000/generate \
    -H 'Content-Type: application/json' \
    -d '{"prompt":"你好，介绍一下你自己","max_tokens":64,"lora_path":"/path/to/lora","adapter_name":"my_lora"}'
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import time
import uuid


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Raw prompt text to generate from")
    # Common sampling params
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # LoRA controls (optional)
    lora_path: Optional[str] = Field(
        default=None,
        description="Filesystem path to a LoRA adapter directory (PEFT/HF).",
    )
    adapter_name: Optional[str] = Field(
        default=None,
        description="Logical adapter name for the LoRA. If omitted, a name is derived from the path.",
    )


class GenerateResponse(BaseModel):
    text: str
    finish_reason: Optional[str] = None
    model: str
    used_lora: Optional[str] = None


class VLLMService:
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        enable_lora: bool = True,
        max_model_len: Optional[int] = None,
        served_model_name: Optional[str] = None,
        default_lora_path: Optional[str] = None,
        default_adapter_name: Optional[str] = None,
    ) -> None:
        self.model_id = model
        self.served_model_name = served_model_name or os.path.basename(model)
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=enable_lora,
            max_model_len=max_model_len,
        )
        # Default LoRA (applied when request does not specify one)
        self.default_lora_request: Optional[LoRARequest] = None
        self.default_used_lora_name: Optional[str] = None
        if default_lora_path:
            if not os.path.isdir(default_lora_path):
                raise ValueError(f"Default LoRA path not found: {default_lora_path}")
            name = default_adapter_name or self._stable_adapter_name(default_lora_path)
            adapter_id = self._adapter_id_from_path(default_lora_path)
            self.default_lora_request = self._make_lora_request(name, adapter_id, default_lora_path)
            self.default_used_lora_name = name

    @staticmethod
    def _stable_adapter_name(path: str) -> str:
        # Derive a short logical name from path for convenience
        base = os.path.basename(os.path.abspath(path))
        h = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]
        return f"{base}-{h}"

    @staticmethod
    def _adapter_id_from_path(path: str) -> int:
        # Derive a stable positive 31-bit adapter_id from path
        h = hashlib.md5(path.encode("utf-8")).hexdigest()
        return int(h[:8], 16) & 0x7FFFFFFF

    @staticmethod
    def _make_lora_request(adapter_name: str, adapter_id: int, lora_path: str) -> LoRARequest:
        # Build a LoRARequest compatible with multiple vLLM versions.
        try:
            sig = inspect.signature(LoRARequest)
            params = set(sig.parameters.keys())
        except Exception:
            params = set()

        # Try kwargs based on detected param names
        kwargs_options = []
        if params:
            kw = {}
            if 'adapter_name' in params:
                kw['adapter_name'] = adapter_name
            elif 'lora_name' in params:
                kw['lora_name'] = adapter_name
            elif 'name' in params:
                kw['name'] = adapter_name

            if 'adapter_id' in params:
                kw['adapter_id'] = adapter_id
            elif 'lora_int_id' in params:
                kw['lora_int_id'] = adapter_id
            elif 'lora_id' in params:
                kw['lora_id'] = adapter_id

            if 'lora_path' in params:
                kw['lora_path'] = lora_path
            elif 'path' in params:
                kw['path'] = lora_path

            kwargs_options.append(kw)

        for kw in kwargs_options:
            try:
                return LoRARequest(**kw)
            except TypeError:
                pass

        # Fallback positional guesses across versions
        positional_guesses = [
            (adapter_name, adapter_id, lora_path),
            (lora_path, adapter_name, adapter_id),
            (adapter_name, lora_path),
            (lora_path, adapter_name),
            (lora_path,),
        ]
        last_err = None
        for args in positional_guesses:
            try:
                return LoRARequest(*args)
            except TypeError as e:
                last_err = e
                continue
        raise TypeError(f"LoRARequest incompatible signature. Tried variants; last error: {last_err}")

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        sampling = SamplingParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            stop=req.stop,
            presence_penalty=req.presence_penalty,
            frequency_penalty=req.frequency_penalty,
        )

        lora_request = None
        used_lora_name: Optional[str] = None

        if req.lora_path:
            if not os.path.isdir(req.lora_path):
                raise HTTPException(status_code=400, detail=f"LoRA path not found: {req.lora_path}")
            used_lora_name = req.adapter_name or self._stable_adapter_name(req.lora_path)
            adapter_id = self._adapter_id_from_path(req.lora_path)
            # Per-request LoRA application
            lora_request = self._make_lora_request(used_lora_name, adapter_id, req.lora_path)

        # Use default LoRA when no per-request LoRA provided
        if lora_request is None and self.default_lora_request is not None:
            lora_request = self.default_lora_request
            used_lora_name = self.default_used_lora_name

        # Generate
        outputs = self.llm.generate(
            [req.prompt],
            sampling,
            lora_request=lora_request,
        )

        out = outputs[0]
        text = out.outputs[0].text if out.outputs else ""
        finish = out.outputs[0].finish_reason if out.outputs else None
        return GenerateResponse(text=text, finish_reason=finish, model=self.model_id, used_lora=used_lora_name)


def build_app(service: VLLMService) -> FastAPI:
    app = FastAPI(title="vLLM + LoRA Server", version="0.1.0")

    @app.get("/")
    def root():
        return {
            "status": "ok",
            "model": service.model_id,
            "message": "POST /generate with {prompt,...} to get completions.",
        }

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest):
        return service.generate(req)

    # ------------------ OpenAI-compatible endpoints under /v1 ------------------
    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": service.served_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "owner",
                }
            ],
        }

    def _map_openai_sampling(body: dict) -> SamplingParams:
        return SamplingParams(
            max_tokens=body.get("max_tokens", body.get("max_new_tokens", 2048)),
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.95),
            top_k=body.get("top_k", 50),
            stop=body.get("stop"),
            presence_penalty=body.get("presence_penalty", 0.0),
            frequency_penalty=body.get("frequency_penalty", 0.0),
            repetition_penalty=body.get("repetition_penalty", 1.0),
        )

    def _extract_lora_from_openai(body: dict):
        extra = body.get("extra_body", {}) or {}
        lora_path = body.get("lora_path") or extra.get("lora_path")
        adapter_name = body.get("adapter_name") or extra.get("adapter_name")
        return lora_path, adapter_name

    def _maybe_build_lora_request(lora_path: Optional[str], adapter_name: Optional[str]):
        if not lora_path:
            return None, None
        if not os.path.isdir(lora_path):
            raise HTTPException(status_code=400, detail=f"LoRA path not found: {lora_path}")
        name = adapter_name or VLLMService._stable_adapter_name(lora_path)
        adapter_id = VLLMService._adapter_id_from_path(lora_path)
        return VLLMService._make_lora_request(name, adapter_id, lora_path), name

    def _calc_usage(out, prompt: str, tokenizer):
        try:
            inp = len(getattr(out, "prompt_token_ids"))  # vLLM provides this
        except Exception:
            try:
                inp = len(tokenizer(prompt, add_special_tokens=False).input_ids) if tokenizer else 0
            except Exception:
                inp = 0
        try:
            comp = sum(len(seq.token_ids) for seq in (out.outputs or []))
        except Exception:
            comp = 0
        total = inp + comp
        return inp, comp, total

    @app.post("/v1/completions")
    async def oai_completions(req: Request):
        body = await req.json()
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            if not prompt:
                raise HTTPException(status_code=400, detail="Empty prompt list")
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise HTTPException(status_code=400, detail="`prompt` must be a string or single-item list")

        sampling = _map_openai_sampling(body)
        lora_path, adapter_name = _extract_lora_from_openai(body)
        lora_request, used_name = _maybe_build_lora_request(lora_path, adapter_name)

        # Offload blocking generate to thread to enable concurrency and batching
        import anyio
        outputs = await anyio.to_thread.run_sync(
            lambda: service.llm.generate([prompt], sampling, lora_request=lora_request)
        )
        out = outputs[0]
        text = out.outputs[0].text if out.outputs else ""
        finish = out.outputs[0].finish_reason if out.outputs else None
        # usage
        try:
            tokenizer = service.llm.get_tokenizer()
        except Exception:
            tokenizer = None
        inp, comp, total = _calc_usage(out, prompt, tokenizer)
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": service.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": finish or "stop",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": int(inp),
                "completion_tokens": int(comp),
                "total_tokens": int(total),
            },
            "x_used_lora": used_name,
        }

    def _build_chat_prompt(messages: List[dict]) -> str:
        # Try to use tokenizer chat template if available
        try:
            tokenizer = service.llm.get_tokenizer()
        except Exception:
            tokenizer = None
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        # Fallback: naive role-tag concatenation
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    @app.post("/v1/chat/completions")
    async def oai_chat_completions(req: Request):
        body = await req.json()
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")

        prompt = _build_chat_prompt(messages)
        sampling = _map_openai_sampling(body)
        lora_path, adapter_name = _extract_lora_from_openai(body)
        lora_request, used_name = _maybe_build_lora_request(lora_path, adapter_name)

        # Offload blocking generate to thread to enable concurrency and batching
        import anyio
        outputs = await anyio.to_thread.run_sync(
            lambda: service.llm.generate([prompt], sampling, lora_request=lora_request)
        )
        out = outputs[0]
        text = out.outputs[0].text if out.outputs else ""
        finish = out.outputs[0].finish_reason if out.outputs else None
        try:
            tokenizer = service.llm.get_tokenizer()
        except Exception:
            tokenizer = None
        inp, comp, total = _calc_usage(out, prompt, tokenizer)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": service.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": int(inp),
                "completion_tokens": int(comp),
                "total_tokens": int(total),
            },
            "x_used_lora": used_name,
        }

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Serve a base model with optional LoRA via vLLM")
    p.add_argument("--model", required=True, help="HF model ID or local path to the base model")
    p.add_argument("--tokenizer", default=None, help="Optional tokenizer ID/path (defaults to model)")
    p.add_argument("--dtype", default="auto", help="Model dtype: auto, float16, bfloat16, float32")
    p.add_argument("--trust-remote-code", action="store_true", help="Allow custom modeling code from HF repos")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--gpu-mem-util", type=float, default=0.9, help="GPU memory utilization fraction")
    p.add_argument("--max-model-len", type=int, default=None, help="Optional max model length override")
    p.add_argument("--default-lora-path", default=None, help="默认启用的LoRA目录（所有请求都应用，除非请求指定其他LoRA）")
    p.add_argument("--default-adapter-name", default=None, help="默认LoRA的适配器名称（可不填）")
    p.add_argument("--served-model-name", default=None, help="OpenAI接口返回中的模型名（默认取模型目录名）")
    p.add_argument("--host", default="127.0.0.1", help="Server host")
    p.add_argument("--port", type=int, default=8801, help="Server port (默认 8801 以匹配文档示例)")
    return p.parse_args()


def main():
    args = parse_args()
    service = VLLMService(
        model=args.model,
        tokenizer=args.tokenizer,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_lora=True,
        max_model_len=args.max_model_len,
        served_model_name=args.served_model_name,
        default_lora_path=args.default_lora_path,
        default_adapter_name=args.default_adapter_name,
    )

    app = build_app(service)

    # Import here so the dependency is optional for non-server usage contexts
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()




"""
- 启动服务（以 Qwen3-4B 为例，端口为 8801）:
      - 基础模型：
CUDA_VISIBLE_DEVICES=0,1,2,3 python serve_vllm_lora.py --model Qwen/Qwen3-4B --served-model-name qwen3.4b --trust-remote-code --tp 4 --port 8801
evalscope eval --model qwen3.4b --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets gsm8k --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":4096}' --repeats 3 --limit 3
      - 启用opd LoRA:
CUDA_VISIBLE_DEVICES=0,1,2,3 python serve_vllm_lora.py --model Qwen/Qwen3-4B --served-model-name qwen3.4b.opd500 --trust-remote-code --tp 4 --port 8801 --default-lora-path ~/Documents/Dual-KL-Distillation/out/opd-4b-32b-deepmath-long/step-500 --default-adapter-name opd500
evalscope eval --model qwen3.4b.opd500 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets gsm8k --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":4096}' --repeats 3 --limit 3
      - 启用dkl LoRA:
CUDA_VISIBLE_DEVICES=0,1,2,3 python serve_vllm_lora.py --model Qwen/Qwen3-4B --served-model-name qwen3.4b.dkl500 --trust-remote-code --tp 4 --port 8801 --default-lora-path ~/Documents/Dual-KL-Distillation/out/dkl-4b-32b-deepmath-long/step-500 --default-adapter-name dkl500
evalscope eval --model qwen3.4b.dkl500 --api-url http://127.0.0.1:8801/v1 --api-key EMPTY --eval-type openai_api --datasets gsm8k --generation-config '{"do_sample":true,"temperature":0.7,"max_tokens":4096}' --repeats 3 --limit 3

- 健康检查：
curl http://127.0.0.1:8801/v1/models

"""