"""RunMux ComfyUI custom nodes.

These nodes call the RunMux **cloud** API through the official Python SDK
(``runmux``). There is no local GPU work — the heavy lifting happens on RunMux's
servers and you get back a video URL (and, optionally, a downloaded mp4).

Install the SDK first (see this pack's README), then drop the nodes in, fill the
fields, and run.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

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
_MODELS = ["seedance-2-0-mini", "seedance-2-0"]
_RESOLUTIONS = ["480p", "720p", "1080p"]


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


class RunMuxGenerateVideo:
    """Generate a video from a prompt (and optionally an image) via RunMux.

    Returns the downloadable video URL, and — when ``download`` is on — also
    saves the mp4 into ComfyUI's output directory and returns that path.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "seedance-2-0-mini"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "resolution": (_RESOLUTIONS, {"default": "480p"}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 60, "step": 1}),
                "auto_enroll_faces": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_url": ("STRING", {"default": "", "multiline": False}),
                "download": ("BOOLEAN", {"default": True}),
            },
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
        image_url: str = "",
        download: bool = True,
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
        image_url = (image_url or "").strip()
        if image_url:
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


class RunMuxEnrollFace:
    """Enroll a (non-celebrity) face photo into the RunMux asset library.

    Returns an ``asset://<id>`` reference you can paste into the
    'RunMux Generate Video' node's ``image_url`` field.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
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
        if not url and not data_url:
            raise RuntimeError(
                "Provide either 'url' (a public https image URL) or 'data_url' "
                "(an inline data: image) to enroll a face."
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
    "RunMuxEnrollFace": RunMuxEnrollFace,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunMuxGenerateVideo": "RunMux Generate Video",
    "RunMuxEnrollFace": "RunMux Enroll Face",
}
