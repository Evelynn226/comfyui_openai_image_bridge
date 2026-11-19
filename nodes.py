from __future__ import annotations

import base64
import io
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image, ImageOps

try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover - standalone/testing fallback
    folder_paths = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _json_object(raw: str, label: str) -> Dict[str, Any]:
    if not raw or not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} 不是有效 JSON：{exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON 对象，例如 {{\"key\": \"value\"}}")
    return value


def _clean_base_url(base_url: str) -> str:
    value = (base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("API Base URL 不能为空。")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("API Base URL 必须以 http:// 或 https:// 开头。")
    return value


def _endpoint_url(base_url: str, endpoint: str) -> str:
    """Accept either a server root or an OpenAI-style /v1 base URL."""
    base = _clean_base_url(base_url)
    suffix = endpoint.lstrip("/")
    if base.lower().endswith("/" + suffix.lower()):
        return base
    if re.search(r"/v\d+(?:/.*)?$", base, flags=re.IGNORECASE):
        return f"{base}/{suffix}"
    return f"{base}/v1/{suffix}"


def _comfy_output_dir() -> Path:
    if folder_paths is not None:
        try:
            return Path(folder_paths.get_output_directory()) / "openai_image_bridge"
        except Exception:
            pass
    # Fallback for import-time tests or non-standard ComfyUI launchers.
    return Path.cwd() / "output" / "openai_image_bridge"


def _safe_prefix(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", (value or "ai_image").strip())
    value = value.strip("._-")
    return (value or "ai_image")[:80]


def _tensor_batch_to_pngs(image_tensor: torch.Tensor) -> List[bytes]:
    """Convert a ComfyUI IMAGE tensor (BHWC or HWC, float 0..1) to PNG bytes."""
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.as_tensor(image_tensor)
    tensor = image_tensor.detach().to(device="cpu", dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"输入 IMAGE 维度应为 HWC 或 BHWC，实际为 {tuple(tensor.shape)}")
    if tensor.shape[-1] not in (1, 3, 4):
        raise ValueError(f"输入 IMAGE 通道数应为 1、3 或 4，实际为 {tensor.shape[-1]}")

    results: List[bytes] = []
    for frame in tensor:
        arr = (frame.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        if arr.shape[-1] == 1:
            pil = Image.fromarray(arr[..., 0], mode="L").convert("RGB")
        elif arr.shape[-1] == 4:
            pil = Image.fromarray(arr, mode="RGBA")
        else:
            pil = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        results.append(buf.getvalue())
    return results


def _mask_to_png(mask_tensor: torch.Tensor, target_size: Tuple[int, int]) -> bytes:
    """Convert ComfyUI MASK (1=selected) to an edit mask (transparent=edit area)."""
    mask = torch.as_tensor(mask_tensor).detach().to(device="cpu", dtype=torch.float32)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"MASK 维度应为 HW 或 BHW，实际为 {tuple(mask.shape)}")
    mask_arr = (mask.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    alpha = 255 - mask_arr
    rgba = np.full((alpha.shape[0], alpha.shape[1], 4), 255, dtype=np.uint8)
    rgba[..., 3] = alpha
    pil = Image.fromarray(rgba, mode="RGBA")
    if pil.size != target_size:
        pil = pil.resize(target_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _decode_data_uri(value: str) -> Optional[Tuple[bytes, str]]:
    match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", value.strip(), flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    mime = (match.group(1) or "application/octet-stream").lower()
    data = match.group(3)
    try:
        raw = base64.b64decode(data) if match.group(2) else requests.utils.unquote(data).encode("utf-8")
    except Exception:
        return None
    return raw, mime


def _extension_from_mime(mime: str, default: str = "png") -> str:
    mime = (mime or "").lower().split(";")[0].strip()
    table = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/svg+xml": "svg",
    }
    return table.get(mime, default)


def _fetch_url_image(url: str, timeout: int) -> Tuple[bytes, str]:
    # Deliberately do not forward the API Authorization header to an arbitrary
    # host returned by the provider; signed URLs should be independently usable.
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    mime = response.headers.get("Content-Type", "application/octet-stream")
    return response.content, mime


def _append_candidate(candidates: List[Tuple[bytes, str]], value: Any, default_format: str, timeout: int) -> None:
    if isinstance(value, str):
        data_uri = _decode_data_uri(value)
        if data_uri:
            candidates.append(data_uri)
            return
        # Some providers put the base64 image in a plain string field.
        compact = value.strip()
        if len(compact) > 32 and re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", compact):
            try:
                candidates.append((base64.b64decode(compact), f"image/{default_format}"))
                return
            except Exception:
                pass
        if compact.startswith("https://") or compact.startswith("http://"):
            try:
                candidates.append(_fetch_url_image(compact, timeout))
            except Exception as exc:
                raise RuntimeError(f"下载接口返回的图片 URL 失败：{exc}") from exc
        return

    if isinstance(value, dict):
        # OpenAI Images API format.
        for key in ("b64_json", "base64", "data", "image_base64", "svg"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                if key == "svg" and not candidate.startswith("data:") and "<svg" in candidate[:500].lower():
                    candidates.append((candidate.encode("utf-8"), "image/svg+xml"))
                    return
                before = len(candidates)
                _append_candidate(candidates, candidate, default_format, timeout)
                if len(candidates) > before:
                    return
        for key in ("url", "image_url"):
            candidate = value.get(key)
            if isinstance(candidate, dict):
                candidate = candidate.get("url")
            if isinstance(candidate, str):
                before = len(candidates)
                _append_candidate(candidates, candidate, default_format, timeout)
                if len(candidates) > before:
                    return


def _extract_images(payload: Dict[str, Any], default_format: str, timeout: int) -> List[Tuple[bytes, str]]:
    candidates: List[Tuple[bytes, str]] = []

    data_items = payload.get("data")
    if isinstance(data_items, list):
        for item in data_items:
            _append_candidate(candidates, item, default_format, timeout)

    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or choice.get("delta") or {}
            if not isinstance(message, dict):
                continue
            images = message.get("images")
            if isinstance(images, list):
                for item in images:
                    _append_candidate(candidates, item, default_format, timeout)
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    _append_candidate(candidates, item, default_format, timeout)
            elif isinstance(content, str):
                # Some OpenAI-compatible bridges embed data URIs in text content.
                for match in re.findall(r"data:image/[^\s)\"']+", content):
                    _append_candidate(candidates, match.rstrip(".,]"), default_format, timeout)

    # A few provider adapters use top-level output / images collections.
    for key in ("images", "output"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                _append_candidate(candidates, item, default_format, timeout)
        elif isinstance(value, dict):
            _append_candidate(candidates, value, default_format, timeout)

    # De-duplicate byte-identical returns.
    unique: List[Tuple[bytes, str]] = []
    seen = set()
    for raw, mime in candidates:
        marker = hash(raw)
        if marker not in seen:
            seen.add(marker)
            unique.append((raw, mime))
    return unique


def _svg_to_png(raw: bytes) -> bytes:
    try:
        import cairosvg  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "收到 SVG 文件，但当前 Python 环境未安装 CairoSVG。请在 ComfyUI 使用的 Python 环境执行：pip install cairosvg"
        ) from exc
    try:
        return cairosvg.svg2png(bytestring=raw)
    except Exception as exc:
        raise RuntimeError(f"SVG 栅格化预览失败（原始 SVG 仍会保存在输出目录）：{exc}") from exc


def _raw_to_pil(raw: bytes, ext: str) -> Image.Image:
    if ext == "svg":
        raw = _svg_to_png(raw)
    try:
        image = Image.open(io.BytesIO(raw))
        image.seek(0)  # Animated GIF: expose its first frame to the ComfyUI IMAGE socket.
        image.load()
        return image.copy()
    except Exception as exc:
        raise RuntimeError(f"无法读取生成的 {ext.upper()} 图片：{exc}") from exc


def _pil_to_comfy(pil_images: Sequence[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
    if not pil_images:
        raise ValueError("API 没有返回可识别的图片数据。")

    rgba_list: List[Image.Image] = [img.convert("RGBA") for img in pil_images]
    max_w = max(img.width for img in rgba_list)
    max_h = max(img.height for img in rgba_list)

    # IMAGE batches require equal dimensions. Pad (without stretching) only if a
    # provider unexpectedly returns different sizes in one response.
    rgb_items: List[np.ndarray] = []
    mask_items: List[np.ndarray] = []
    for img in rgba_list:
        if img.size != (max_w, max_h):
            canvas = Image.new("RGBA", (max_w, max_h), (255, 255, 255, 0))
            x = (max_w - img.width) // 2
            y = (max_h - img.height) // 2
            canvas.alpha_composite(img, (x, y))
            img = canvas
        rgba = np.asarray(img, dtype=np.uint8)
        rgb = rgba[..., :3].astype(np.float32) / 255.0
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        rgb_items.append(rgb)
        # ComfyUI masks conventionally use 1 for the masked/transparent area.
        mask_items.append(1.0 - alpha)

    images = torch.from_numpy(np.stack(rgb_items, axis=0).astype(np.float32))
    masks = torch.from_numpy(np.stack(mask_items, axis=0).astype(np.float32))
    return images, masks


def _response_error(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error", payload)
            if isinstance(error, dict):
                message = error.get("message") or error.get("detail") or error.get("type")
                if message:
                    return str(message)
            return json.dumps(payload, ensure_ascii=False)[:2000]
    except Exception:
        pass
    return (response.text or f"HTTP {response.status_code}")[:2000]


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------


class OpenAICompatibleAPIConfig:
    """Reusable connection settings for one or more generation nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": "https://api.openai.com/v1", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "留空时读取环境变量 OPENAI_COMPAT_API_KEY / OPENAI_API_KEY"}),
                "timeout_seconds": ("INT", {"default": 300, "min": 10, "max": 1800, "step": 10}),
                "verify_ssl": ("BOOLEAN", {"default": True}),
                "extra_headers_json": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("OPENAI_COMPAT_CONFIG",)
    RETURN_NAMES = ("api_config",)
    FUNCTION = "configure"
    CATEGORY = "AI 图片生成/OpenAI 兼容"

    def configure(self, base_url, api_key, timeout_seconds, verify_ssl, extra_headers_json):
        config = {
            "base_url": _clean_base_url(base_url),
            "api_key": api_key.strip(),
            "timeout": int(timeout_seconds),
            "verify_ssl": bool(verify_ssl),
            "extra_headers": _json_object(extra_headers_json, "额外请求头"),
        }
        return (config,)


class OpenAICompatibleImageGenerate:
    """Generate images through an OpenAI-compatible Images or Chat Completions API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_config": ("OPENAI_COMPAT_CONFIG",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": ("STRING", {"default": "gpt-image-2.6-sunburst", "multiline": False}),
                "api_mode": (["Images API", "Chat Completions"], {"default": "Images API"}),
                "output_format": (["png", "jpeg", "webp", "gif", "svg"], {"default": "png"}),
                "size": ("STRING", {"default": "auto", "multiline": False}),
                "quality": (["auto", "low", "medium", "high", "xhigh", "max"], {"default": "auto"}),
                "background": (["auto", "opaque", "transparent"], {"default": "auto"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "output_compression": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "frame_count": ("INT", {"default": 9, "min": 2, "max": 16, "step": 1}),
                "frame_duration_ms": ("INT", {"default": 200, "min": 20, "max": 10000, "step": 10}),
                "complexity": (["low", "medium", "high"], {"default": "medium"}),
                "svg_model": ("STRING", {"default": "", "multiline": False, "placeholder": "可留空；仅 SVG 模式且服务商开放 SVG 模型选择时填写"}),
                "svg_reasoning": (["none", "minimal", "low", "medium", "high", "xhigh", "max"], {"default": "medium"}),
                "send_modalities": ("BOOLEAN", {"default": True}),
                "send_image_options": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "ai_image", "multiline": False}),
                "extra_parameters_json": ("STRING", {"default": "{}", "multiline": True}),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "saved_files")
    FUNCTION = "generate"
    CATEGORY = "AI 图片生成/OpenAI 兼容"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Never reuse a previous generation result from ComfyUI's execution cache.

        Image generation is an action: even when prompt and settings are unchanged,
        queueing the workflow again must issue a fresh API request. NaN is treated
        by ComfyUI as always changed.
        """
        return float("NaN")

    def generate(
        self,
        api_config,
        prompt,
        model,
        api_mode,
        output_format,
        size,
        quality,
        background,
        n,
        output_compression,
        frame_count,
        frame_duration_ms,
        complexity,
        svg_model,
        svg_reasoning,
        send_modalities,
        send_image_options,
        filename_prefix,
        extra_parameters_json,
        input_image=None,
        mask=None,
    ):
        if not isinstance(api_config, dict):
            raise ValueError("请先连接「OpenAI 兼容 API 配置」节点。")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt 不能为空。")
        model = (model or "").strip()
        if not model:
            raise ValueError("Model 名称不能为空。")

        base_url = _clean_base_url(api_config.get("base_url", ""))
        api_key = (api_config.get("api_key") or os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
        timeout = int(api_config.get("timeout", 300))
        verify_ssl = bool(api_config.get("verify_ssl", True))
        headers = {"Accept": "application/json"}
        extra_headers = api_config.get("extra_headers") or {}
        if not isinstance(extra_headers, dict):
            raise ValueError("API 配置节点中的 extra_headers_json 必须是 JSON 对象。")
        headers.update({str(k): str(v) for k, v in extra_headers.items()})
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        extra_parameters = _json_object(extra_parameters_json, "额外请求参数")
        fmt = output_format.lower().strip()
        if fmt not in {"png", "jpeg", "webp", "gif", "svg"}:
            raise ValueError(f"不支持的输出格式：{output_format}")
        if fmt in ("gif", "svg") and int(n) != 1:
            raise ValueError("GIF / SVG 输出目前要求 n=1。请将 n 设为 1。")
        if fmt == "gif" and mask is not None:
            raise ValueError("GPT Image GIF 输出不支持 mask；请断开 mask 输入。")
        if fmt == "svg" and int(n) != 1:
            raise ValueError("SVG 输出要求 n=1。")

        session = requests.Session()
        session.verify = verify_ssl
        saved_paths: List[str] = []
        raw_results: List[Tuple[bytes, str]]

        if api_mode == "Images API":
            payload: Dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "n": int(n),
                "size": (size or "auto").strip() or "auto",
                "quality": quality,
                "output_format": fmt,
                "background": background,
                "complexity": complexity,
            }
            if fmt in ("jpeg", "webp"):
                payload["output_compression"] = int(output_compression)
            if fmt == "gif":
                payload["frame_count"] = int(frame_count)
                payload["frame_duration_ms"] = int(frame_duration_ms)
            if fmt == "svg" and (svg_model or "").strip():
                payload["svg"] = {
                    "model": svg_model.strip(),
                    "reasoning": {"effort": svg_reasoning},
                }
            payload.update(extra_parameters)

            if input_image is not None:
                url = _endpoint_url(base_url, "images/edits")
                image_blobs = _tensor_batch_to_pngs(input_image)
                files = []
                for index, blob in enumerate(image_blobs):
                    field = "image" if len(image_blobs) == 1 else "image[]"
                    files.append((field, (f"reference_{index + 1}.png", blob, "image/png")))
                if mask is not None:
                    with Image.open(io.BytesIO(image_blobs[0])) as ref_img:
                        target_size = ref_img.size
                    mask_blob = _mask_to_png(mask, target_size)
                    files.append(("mask", ("mask.png", mask_blob, "image/png")))
                # Multipart form: nested object parameters must be JSON strings.
                form_data = {}
                for key, value in payload.items():
                    form_data[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                request_headers = dict(headers)
                request_headers.pop("Content-Type", None)
                response = session.post(url, headers=request_headers, data=form_data, files=files, timeout=timeout)
            else:
                url = _endpoint_url(base_url, "images/generations")
                request_headers = dict(headers)
                request_headers["Content-Type"] = "application/json"
                response = session.post(url, headers=request_headers, json=payload, timeout=timeout)

            if not response.ok:
                raise RuntimeError(f"Images API 请求失败（HTTP {response.status_code}）：{_response_error(response)}")
            try:
                response_payload = response.json()
            except Exception as exc:
                raise RuntimeError(f"Images API 返回的不是 JSON：{(response.text or '')[:1000]}") from exc
            raw_results = _extract_images(response_payload, fmt, timeout)

        elif api_mode == "Chat Completions":
            url = _endpoint_url(base_url, "chat/completions")
            content: Any = [{"type": "text", "text": prompt}]
            if input_image is not None:
                for blob in _tensor_batch_to_pngs(input_image):
                    data_url = "data:image/png;base64," + base64.b64encode(blob).decode("ascii")
                    content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
            message_content: Any = content if input_image is not None else prompt
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": message_content}],
                "stream": False,
            }
            if send_modalities:
                payload["modalities"] = ["text", "image"]
            # Chat Completions has no universal image-format/size schema. Keep
            # these provider-specific hints opt-in to avoid breaking strict
            # OpenAI-compatible gateways. Nano Banana options such as
            # image_config should be supplied through extra_parameters_json.
            if send_image_options:
                payload["output_format"] = fmt
                if (size or "auto").strip().lower() != "auto":
                    payload["size"] = (size or "").strip()
                if quality != "auto":
                    payload["quality"] = quality
            payload.update(extra_parameters)
            request_headers = dict(headers)
            request_headers["Content-Type"] = "application/json"
            response = session.post(url, headers=request_headers, json=payload, timeout=timeout)
            if not response.ok:
                raise RuntimeError(f"Chat Completions 请求失败（HTTP {response.status_code}）：{_response_error(response)}")
            try:
                response_payload = response.json()
            except Exception as exc:
                raise RuntimeError(f"Chat Completions 返回的不是 JSON：{(response.text or '')[:1000]}") from exc
            raw_results = _extract_images(response_payload, fmt, timeout)
        else:
            raise ValueError(f"未知 API 模式：{api_mode}")

        if not raw_results:
            raise RuntimeError(
                "接口请求成功，但没有从响应中找到图片。\n"
                "请确认该服务商支持所选 API 模式，并检查其图片响应格式；"
                "Chat Completions 模式可尝试在 extra_parameters_json 中添加服务商要求的参数。"
            )

        output_dir = _comfy_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = _safe_prefix(filename_prefix)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pil_images: List[Image.Image] = []
        for index, (raw, mime) in enumerate(raw_results, start=1):
            ext = _extension_from_mime(mime, fmt)
            # The requested format is authoritative when a gateway omits MIME
            # information (common with base64-only Images API responses).
            if ext == "bin":
                ext = fmt
            name = f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}_{index}.{ext}"
            path = output_dir / name
            path.write_bytes(raw)
            saved_paths.append(str(path))
            pil_images.append(_raw_to_pil(raw, ext))

        images_tensor, masks_tensor = _pil_to_comfy(pil_images)
        return (images_tensor, masks_tensor, "\n".join(saved_paths))


NODE_CLASS_MAPPINGS = {
    "OpenAICompatibleAPIConfig": OpenAICompatibleAPIConfig,
    "OpenAICompatibleImageGenerate": OpenAICompatibleImageGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatibleAPIConfig": "OpenAI 兼容 API 配置",
    "OpenAICompatibleImageGenerate": "GPT Image / Nano Banana 图片生成",
}
