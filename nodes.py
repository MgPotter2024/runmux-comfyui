"""RunMux ComfyUI custom nodes.

These nodes call the RunMux **cloud** API through the official Python SDK
(``runmux``). There is no local GPU work — the heavy lifting happens on RunMux's
servers and you get back a video URL (and, optionally, a downloaded mp4).

Install the SDK first (see this pack's README), then drop the nodes in, fill the
fields, and run.

Nodes:
  - RunMux Generate Video — text/image(s) to video. Wire up to 9 reference
    images straight from ComfyUI "Load Image" nodes (``image_1..image_9``),
    or paste ``asset://`` ids / https URLs into ``reference_assets`` (one per
    line; commas ok). ``first_frame``/``last_frame`` inputs give exact frame
    control.
  - RunMux Save Video — download the finished video and decode it into an
    IMAGE frame batch for preview/further processing inside ComfyUI.
  - RunMux Enroll Face — enroll a face photo (IMAGE input, data: URL, or
    public URL) into the RunMux asset library; returns ``asset://<id>``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    # The official RunMux Python SDK. Installed separately (see README).
    import runmux
except ImportError as exc:  # pragma: no cover - exercised only without the SDK
    raise ImportError(
        "The RunMux ComfyUI nodes require the 'runmux' Python SDK.\n"
        "Install it into the same Python environment ComfyUI runs in, e.g.:\n"
        "    pip install runmux\n"
        "or, for local development against this repo:\n"
        "    pip install -e <runmux>/sdk/python"
    ) from exc


# Models the user can pick from in the dropdown. seedance-2-0-mini is the
# default; a free-text "model" string node could be added later if needed.
_MODELS = ["seedance-2-0-mini", "seedance-2-0", "seedance-2-0-fast"]
_RESOLUTIONS = ["480p", "720p", "1080p"]

# Up to 9 reference images — the Seedance 2.0 upstream cap.
_MAX_REFERENCE_IMAGES = 9

# Reference images are re-encoded as JPEG data: URIs before upload. Nine raw
# PNGs would blow past the API's request-body limit; JPEG q90 with the long
# side capped at 2048px keeps quality high and payloads small (the upstream
# model re-processes inputs anyway).
_JPEG_QUALITY = 90
_MAX_SIDE = 2048


def _tensor_to_data_uri(image: Any, max_side: int = _MAX_SIDE, quality: int = _JPEG_QUALITY) -> str:
    """Encode a ComfyUI IMAGE ([B,H,W,C] or [H,W,C], float 0-1) as a JPEG data: URI.

    Accepts torch tensors, numpy arrays, or plain nested lists (used by tests).
    Batches use the first frame. Alpha is dropped (converted to RGB).
    """
    import numpy as np  # lazy: always present inside ComfyUI
    from PIL import Image as PILImage

    t = image
    if hasattr(t, "detach"):  # torch tensor
        t = t.detach().cpu().numpy()
    arr = np.asarray(t)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise RuntimeError(f"Expected an IMAGE of shape [B,H,W,C] or [H,W,C], got {arr.shape!r}")
    arr = (np.clip(arr.astype("float32"), 0.0, 1.0) * 255.0).round().astype("uint8")
    img = PILImage.fromarray(arr)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        img = img.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            PILImage.LANCZOS,
        )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + b64


def _parse_reference_assets(text: str) -> List[str]:
    """Split the reference_assets textarea into individual refs.

    One reference per line; commas also act as separators (the customer's
    first instinct was comma-separated asset ids — honor it).
    """
    out: List[str] = []
    for line in (text or "").replace(",", "\n").splitlines():
        item = line.strip()
        if item:
            out.append(item)
    return out


def _result_url(result: Any) -> str:
    """Pull a downloadable video URL out of a videos.run() result.

    ``run()`` returns a single finished job dict (with ``url`` set) for one
    result, or a list of finished jobs for a batch. We use a single result here,
    so take the first if a list comes back.
    """
    if isinstance(result, list):
        if not result:
            raise RuntimeError("RunMux returned an empty result set.")
        result = result[0]
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected RunMux result type: {type(result)!r}")
    url = result.get("url")
    if not url:
        raise RuntimeError(
            "RunMux finished the job but returned no video URL. "
            f"Raw result: {result!r}"
        )
    return url


def _comfy_output_dir() -> str:
    """Best-effort path to ComfyUI's output directory.

    Falls back to the current working directory if ComfyUI's folder_paths
    helper is not importable (e.g. when running the structural smoke test
    outside ComfyUI).
    """
    try:
        import folder_paths  # type: ignore

        return folder_paths.get_output_directory()
    except Exception:
        return os.getcwd()


def _download(url: str, dest_dir: str, basename: str) -> str:
    """Download ``url`` into ``dest_dir`` and return the saved file path."""
    # httpx is a dependency of the runmux SDK, so it is always available here.
    import httpx

    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, basename)
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(path, "wb") as fh:
            for chunk in resp.iter_bytes():
                fh.write(chunk)
    return path


def _want_frame(index: int, skip_first_frames: int, select_every_nth: int) -> bool:
    """Pure frame-selection predicate shared by both decoders (tested directly)."""
    if index < skip_first_frames:
        return False
    nth = max(1, int(select_every_nth))
    return (index - skip_first_frames) % nth == 0


def _decode_frames(path: str, skip_first_frames: int, frame_load_cap: int, select_every_nth: int):
    """Decode ``path`` into a float32 [N,H,W,C] tensor of RGB frames (0-1).

    Prefers PyAV (bundled with current ComfyUI), falls back to OpenCV.
    """
    import numpy as np

    frames: List[Any] = []

    def _push(rgb_array: Any) -> bool:
        frames.append(np.asarray(rgb_array, dtype=np.float32) / 255.0)
        return bool(frame_load_cap) and len(frames) >= frame_load_cap

    decoded = False
    try:
        import av  # type: ignore

        with av.open(path) as container:
            index = 0
            for frame in container.decode(video=0):
                if _want_frame(index, skip_first_frames, select_every_nth):
                    if _push(frame.to_ndarray(format="rgb24")):
                        break
                index += 1
        decoded = True
    except ImportError:
        pass

    if not decoded:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "解码视频需要 av 或 opencv。请在 ComfyUI 的 Python 环境执行:\n"
                "    pip install av\n"
                "(或 pip install opencv-python)"
            ) from exc
        cap = cv2.VideoCapture(path)
        try:
            index = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if _want_frame(index, skip_first_frames, select_every_nth):
                    if _push(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)):
                        break
                index += 1
        finally:
            cap.release()

    if not frames:
        raise RuntimeError(f"视频解码不到任何帧: {path}")

    import torch  # ComfyUI always ships torch

    return torch.from_numpy(np.stack(frames, axis=0))


class RunMuxGenerateVideo:
    """Generate a video from a prompt and up to 9 reference images via RunMux.

    Reference images can come from ComfyUI "Load Image" nodes (``image_1..9``)
    and/or the ``reference_assets`` textarea (``asset://<id>`` from the RunMux
    asset library, or public https URLs — one per line, commas ok). They are
    merged in that order, 9 total max.

    ``first_frame``/``last_frame`` give exact start/end-frame control, but the
    upstream model treats frame control and reference images as mutually
    exclusive: in reference mode, point at your start frame in the prompt
    (e.g. "以图1为首帧 / use Image 1 as the start frame").
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        optional: Dict[str, Any] = {
            "reference_assets": ("STRING", {
                "default": "",
                "multiline": True,
                "tooltip": "asset://素材库引用 或 图片URL,每行一个(逗号分隔也可)。与 image_1..9 合并,总共最多 9 张。",
            }),
            "image_url": ("STRING", {"default": "", "multiline": False}),
            "download": ("BOOLEAN", {"default": True}),
            "first_frame": ("IMAGE",),
            "last_frame": ("IMAGE",),
        }
        for i in range(1, _MAX_REFERENCE_IMAGES + 1):
            optional[f"image_{i}"] = ("IMAGE",)
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "seedance-2-0-mini"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "resolution": (_RESOLUTIONS, {"default": "480p"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                "auto_enroll_faces": ("BOOLEAN", {"default": False}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_url", "file_path")
    FUNCTION = "generate"
    CATEGORY = "RunMux"

    def generate(
        self,
        api_key: str,
        model: str,
        prompt: str,
        resolution: str,
        duration: int,
        auto_enroll_faces: bool,
        reference_assets: str = "",
        image_url: str = "",
        download: bool = True,
        first_frame: Optional[Any] = None,
        last_frame: Optional[Any] = None,
        **image_inputs: Any,
    ) -> Tuple[str, str]:
        api_key = (api_key or "").strip() or os.environ.get("RUNMUX_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "RunMux API key is required. Paste it into the 'api_key' field "
                "or set the RUNMUX_API_KEY environment variable."
            )
        if not (prompt or "").strip():
            raise RuntimeError("A 'prompt' is required to generate a video.")

        kwargs: Dict[str, Any] = {
            "model": (model or "seedance-2-0-mini").strip(),
            "prompt": prompt,
            "resolution": resolution,
            "duration": int(duration),
            "auto_enroll_faces": bool(auto_enroll_faces),
        }

        # --- collect reference images (wired IMAGE inputs first, in slot order) ---
        references: List[str] = []
        for i in range(1, _MAX_REFERENCE_IMAGES + 1):
            img = image_inputs.get(f"image_{i}")
            if img is not None:
                references.append(_tensor_to_data_uri(img))
        references.extend(_parse_reference_assets(reference_assets))

        image_url = (image_url or "").strip()

        # --- frame-control mode (first/last frame) ---
        frame_images: List[Dict[str, Any]] = []
        if first_frame is not None:
            frame_images.append({"url": _tensor_to_data_uri(first_frame), "frame": "first"})
        if last_frame is not None:
            frame_images.append({"url": _tensor_to_data_uri(last_frame), "frame": "last"})

        if frame_images and (references or image_url):
            raise RuntimeError(
                "首帧/尾帧(frame 控制)与参考图(image_1..9 / reference_assets / image_url)不能同时使用 — "
                "上游模型二选一。要在多参考图模式下指定首帧,请把首帧图接到某个 image_N,"
                "并在提示词里写「以图N为首帧 / use Image N as the start frame」。"
            )

        if frame_images:
            kwargs["frame_images"] = frame_images
        elif references or image_url:
            if references:
                if image_url:
                    references.append(image_url)
                if len(references) > _MAX_REFERENCE_IMAGES:
                    raise RuntimeError(
                        f"参考图最多 {_MAX_REFERENCE_IMAGES} 张(image_1..9 与 reference_assets、image_url 合计),"
                        f"当前 {len(references)} 张。"
                    )
                kwargs["reference_images"] = references
            else:
                # Legacy single-image path: a lone image_url keeps its historical
                # first-frame-ish semantics upstream — do not silently change it.
                kwargs["image_url"] = image_url

        try:
            with runmux.RunmuxClient(api_key=api_key) as client:
                result = client.videos.run(**kwargs)
        except runmux.RunmuxError as exc:
            # Surface RunMux's own code/request_id for actionable errors.
            detail = f" (code={exc.code}"
            if exc.request_id:
                detail += f", request_id={exc.request_id}"
            detail += ")"
            raise RuntimeError(f"RunMux error: {exc.message}{detail}") from exc

        url = _result_url(result)

        file_path = ""
        if download:
            job_id = result[0].get("id") if isinstance(result, list) else result.get("id")
            basename = f"runmux_{job_id or 'video'}.mp4"
            try:
                file_path = _download(url, _comfy_output_dir(), basename)
            except Exception as exc:  # download is best-effort; URL still returned
                raise RuntimeError(
                    f"Generated the video but failed to download it: {exc}. "
                    f"The video URL is still available: {url}"
                ) from exc

        return (url, file_path)


class RunMuxSaveVideo:
    """Save a RunMux video into ComfyUI's output directory and decode it to frames.

    Feed ``video_url`` (or an already-downloaded ``file_path``) from the
    'RunMux Generate Video' node. Returns the frame batch (IMAGE), the saved
    file path, and the frame count.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "save_subfolder": ("STRING", {"default": "runmux"}),
                "filename_prefix": ("STRING", {"default": "runmux"}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "skip_first_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "select_every_nth": ("INT", {"default": 1, "min": 1, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("frames", "file_path", "frame_count")
    FUNCTION = "save"
    CATEGORY = "RunMux"

    def save(
        self,
        video_url: str,
        file_path: str = "",
        save_subfolder: str = "runmux",
        filename_prefix: str = "runmux",
        frame_load_cap: int = 0,
        skip_first_frames: int = 0,
        select_every_nth: int = 1,
    ) -> Tuple[Any, str, int]:
        video_url = (video_url or "").strip()
        file_path = (file_path or "").strip()

        if file_path and os.path.exists(file_path):
            saved = file_path
        elif video_url:
            dest = os.path.join(_comfy_output_dir(), (save_subfolder or "").strip() or "runmux")
            digest = hashlib.sha1(video_url.encode("utf-8")).hexdigest()[:10]
            basename = f"{(filename_prefix or 'runmux').strip() or 'runmux'}_{digest}.mp4"
            candidate = os.path.join(dest, basename)
            saved = candidate if os.path.exists(candidate) else _download(video_url, dest, basename)
        else:
            raise RuntimeError("请传入 video_url(或已下载的 file_path)。")

        frames = _decode_frames(saved, int(skip_first_frames), int(frame_load_cap), int(select_every_nth))
        return (frames, saved, int(frames.shape[0]))


class RunMuxEnrollFace:
    """Enroll a (non-celebrity) face photo into the RunMux asset library.

    Wire a ComfyUI IMAGE straight in, or provide a public URL / data: URL.
    Returns an ``asset://<id>`` reference for the 'RunMux Generate Video'
    node's ``reference_assets`` field.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "url": ("STRING", {"default": "", "multiline": False}),
                "data_url": ("STRING", {"default": "", "multiline": True}),
                "name": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("asset_uri",)
    FUNCTION = "enroll"
    CATEGORY = "RunMux"

    def enroll(
        self,
        api_key: str,
        image: Optional[Any] = None,
        url: str = "",
        data_url: str = "",
        name: str = "",
    ) -> Tuple[str]:
        api_key = (api_key or "").strip() or os.environ.get("RUNMUX_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "RunMux API key is required. Paste it into the 'api_key' field "
                "or set the RUNMUX_API_KEY environment variable."
            )
        url = (url or "").strip()
        data_url = (data_url or "").strip()
        if image is not None and not data_url:
            # Faces deserve a bit more fidelity than scene references.
            data_url = _tensor_to_data_uri(image, quality=95)
        if not url and not data_url:
            raise RuntimeError(
                "Provide an 'image' input, a 'url' (public https image URL), or a "
                "'data_url' (inline data: image) to enroll a face."
            )

        try:
            with runmux.RunmuxClient(api_key=api_key) as client:
                asset_uri = client.faces.enroll(
                    url=url or None,
                    data_url=data_url or None,
                    name=(name or "").strip() or None,
                )
        except runmux.RunmuxError as exc:
            detail = f" (code={exc.code}"
            if exc.request_id:
                detail += f", request_id={exc.request_id}"
            detail += ")"
            raise RuntimeError(f"RunMux error: {exc.message}{detail}") from exc

        return (asset_uri,)


NODE_CLASS_MAPPINGS = {
    "RunMuxGenerateVideo": RunMuxGenerateVideo,
    "RunMuxSaveVideo": RunMuxSaveVideo,
    "RunMuxEnrollFace": RunMuxEnrollFace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunMuxGenerateVideo": "RunMux Generate Video",
    "RunMuxSaveVideo": "RunMux Save Video",
    "RunMuxEnrollFace": "RunMux Enroll Face",
}
