"""Behavioral tests for multi-reference / first-frame / save-video node upgrades.

No network. The RunMux SDK client is monkeypatched; "IMAGE tensors" are plain
nested lists (numpy handles them the same way ComfyUI's torch tensors are
handled after .numpy()).

Run:
    pip install -e ../../sdk/python numpy pillow pytest
    pytest tests/test_nodes_multiref.py -v
"""

from __future__ import annotations

import base64
import importlib.util
import io
import os
import sys

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
from PIL import Image as PILImage  # noqa: E402

PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_nodes():
    spec = importlib.util.spec_from_file_location(
        "runmux_comfyui_nodes_test", os.path.join(PACK_DIR, "nodes.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


nodes = _load_nodes()


def _fake_image(h=8, w=16, batch=True):
    """A ComfyUI-style IMAGE as nested lists: [B,H,W,C] float 0-1."""
    img = [[[0.5, 0.25, 0.75] for _ in range(w)] for _ in range(h)]
    return [img] if batch else img


class _FakeVideos:
    def __init__(self, sink):
        self._sink = sink

    def run(self, **kwargs):
        self._sink["kwargs"] = kwargs
        return {"id": "vid_test", "url": "https://cdn.example/video.mp4"}


class _FakeFaces:
    def __init__(self, sink):
        self._sink = sink

    def enroll(self, **kwargs):
        self._sink["enroll"] = kwargs
        return "asset://fake123"


class _FakeClient:
    def __init__(self, sink):
        self.videos = _FakeVideos(sink)
        self.faces = _FakeFaces(sink)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def sink(monkeypatch):
    captured = {}
    monkeypatch.setattr(nodes.runmux, "RunmuxClient", lambda api_key: _FakeClient(captured))
    return captured


def _generate(sink_dict, **overrides):
    node = nodes.RunMuxGenerateVideo()
    base = dict(
        api_key="rmx_test",
        model="seedance-2-0-mini",
        prompt="a test prompt",
        resolution="480p",
        ratio="16:9",
        duration=5,
        auto_enroll_faces=False,
        download=False,
    )
    base.update(overrides)
    return node.generate(**base)


# ---------- structure ----------

def test_input_types_have_new_fields():
    req = nodes.RunMuxGenerateVideo.INPUT_TYPES()["required"]
    opt = nodes.RunMuxGenerateVideo.INPUT_TYPES()["optional"]
    assert "ratio" in req
    for i in range(1, 10):
        assert f"image_{i}" in opt and opt[f"image_{i}"][0] == "IMAGE"
    assert "reference_assets" in opt
    assert "first_frame" in opt and opt["first_frame"][0] == "IMAGE"
    assert "last_frame" in opt and opt["last_frame"][0] == "IMAGE"
    for i in range(1, 4):
        assert f"reference_audio_{i}" in opt
        assert f"audio_{i}" in opt and opt[f"audio_{i}"][0] == "AUDIO"


def test_save_video_registered():
    assert "RunMuxSaveVideo" in nodes.NODE_CLASS_MAPPINGS
    rt = nodes.RunMuxSaveVideo.RETURN_TYPES
    assert rt == ("IMAGE", "STRING", "INT")


def test_models_include_fast():
    assert "seedance-2-0-fast" in nodes._MODELS


# ---------- helpers ----------

def test_parse_reference_assets_lines_and_commas():
    text = "asset://a1, asset://a2\nhttps://cdn.example/x.jpg\n\n  asset://a3  "
    assert nodes._parse_reference_assets(text) == [
        "asset://a1", "asset://a2", "https://cdn.example/x.jpg", "asset://a3",
    ]


def test_tensor_to_data_uri_jpeg_and_downscale():
    big = np.zeros((1, 10, 4096, 3), dtype=np.float32)  # long side 4096 -> capped to 2048
    uri = nodes._tensor_to_data_uri(big)
    assert uri.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    img = PILImage.open(io.BytesIO(raw))
    assert max(img.size) <= nodes._MAX_SIDE


def test_want_frame_math():
    # skip 2, every 2nd: indices 2,4,6,...
    picked = [i for i in range(10) if nodes._want_frame(i, 2, 2)]
    assert picked == [2, 4, 6, 8]
    # defaults: everything
    assert [i for i in range(4) if nodes._want_frame(i, 0, 1)] == [0, 1, 2, 3]


# ---------- generate: reference images ----------

def test_multi_reference_merge_order(sink):
    _generate(
        sink,
        image_1=_fake_image(),
        image_3=_fake_image(),
        reference_assets="asset://lib1\nasset://lib2",
    )
    refs = sink["kwargs"]["images"]
    assert len(refs) == 4
    assert refs[0].startswith("data:image/jpeg;base64,")
    assert refs[1].startswith("data:image/jpeg;base64,")
    assert refs[2] == "asset://lib1" and refs[3] == "asset://lib2"
    assert "image_url" not in sink["kwargs"]
    assert "frame_images" not in sink["kwargs"]
    assert "reference_images" not in sink["kwargs"]


def test_image_url_merges_into_references_when_refs_present(sink):
    _generate(sink, reference_assets="asset://a", image_url="https://cdn.example/b.jpg")
    assert sink["kwargs"]["images"] == ["asset://a", "https://cdn.example/b.jpg"]


def test_legacy_single_image_url_stays_single(sink):
    _generate(sink, image_url="asset://only-one")
    assert sink["kwargs"]["image_url"] == "asset://only-one"
    assert "reference_images" not in sink["kwargs"]
    assert "images" not in sink["kwargs"]


def test_text_to_video_without_any_image(sink):
    _generate(sink)
    assert "reference_images" not in sink["kwargs"]
    assert "images" not in sink["kwargs"]
    assert "image_url" not in sink["kwargs"]
    assert "frame_images" not in sink["kwargs"]


def test_ratio_passed_to_sdk(sink):
    _generate(sink, ratio="9:16")
    assert sink["kwargs"]["ratio"] == "9:16"


def test_reference_audio_urls_passed_to_sdk(sink):
    _generate(
        sink,
        reference_audio_1="https://cdn.example/a.wav",
        reference_audio_3="https://cdn.example/b.mp3",
    )
    assert sink["kwargs"]["reference_audios"] == [
        "https://cdn.example/a.wav",
        "https://cdn.example/b.mp3",
    ]


def test_reference_audio_local_file_encoded_from_allowed_root(sink, tmp_path, monkeypatch):
    raw = b"RIFF\x24\x00\x00\x00WAVEfmt "
    (tmp_path / "clip.wav").write_bytes(raw)
    monkeypatch.setattr(nodes, "_comfy_audio_roots", lambda: [str(tmp_path)])

    _generate(sink, reference_audio_1="clip.wav")

    ref = sink["kwargs"]["reference_audios"][0]
    assert ref.startswith("data:audio/wav;base64,")
    assert base64.b64decode(ref.split(",", 1)[1]) == raw


def test_reference_audio_native_audio_encoded_to_wav(sink):
    _generate(
        sink,
        audio_1={
            "waveform": np.zeros((1, 1, 16), dtype=np.float32),
            "sample_rate": 16000,
        },
    )
    ref = sink["kwargs"]["reference_audios"][0]
    assert ref.startswith("data:audio/wav;base64,")
    assert base64.b64decode(ref.split(",", 1)[1]).startswith(b"RIFF")


def test_reference_audio_rejects_path_outside_allowed_root(sink, tmp_path, monkeypatch):
    root = tmp_path / "input"
    root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    monkeypatch.setattr(nodes, "_comfy_audio_roots", lambda: [str(root)])

    with pytest.raises(RuntimeError, match="input/temp"):
        _generate(sink, reference_audio_1=str(outside))
    assert "kwargs" not in sink


def test_more_than_nine_references_rejected(sink):
    text = "\n".join(f"asset://a{i}" for i in range(10))
    with pytest.raises(RuntimeError, match="最多 9"):
        _generate(sink, reference_assets=text)


# ---------- generate: frame control ----------

def test_first_and_last_frame_mode(sink):
    _generate(sink, first_frame=_fake_image(), last_frame=_fake_image())
    frames = sink["kwargs"]["frame_images"]
    assert [f["frame"] for f in frames] == ["first", "last"]
    assert all(f["url"].startswith("data:image/jpeg;base64,") for f in frames)
    assert "reference_images" not in sink["kwargs"]
    assert "images" not in sink["kwargs"]


def test_frame_mode_conflicts_with_references(sink):
    with pytest.raises(RuntimeError, match="不能同时使用"):
        _generate(sink, first_frame=_fake_image(), image_1=_fake_image())
    with pytest.raises(RuntimeError, match="不能同时使用"):
        _generate(sink, first_frame=_fake_image(), image_url="asset://x")


# ---------- enroll face ----------

def test_enroll_face_accepts_image_tensor(sink):
    node = nodes.RunMuxEnrollFace()
    (uri,) = node.enroll(api_key="rmx_test", image=_fake_image(), name="试戴模特")
    assert uri == "asset://fake123"
    sent = sink["enroll"]
    assert sent["data_url"].startswith("data:image/jpeg;base64,")
    assert sent["url"] is None
    assert sent["name"] == "试戴模特"


# ---------- save video ----------

def test_save_video_requires_source():
    node = nodes.RunMuxSaveVideo()
    with pytest.raises(RuntimeError, match="video_url"):
        node.save(video_url="", file_path="")


def test_save_video_decodes_local_file(tmp_path):
    av = pytest.importorskip("av")  # decoder optional in dev env; ComfyUI ships it
    torch = pytest.importorskip("torch")
    # synthesize an 8-frame video
    path = str(tmp_path / "clip.mp4")
    with av.open(path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        for i in range(8):
            arr = np.full((32, 64, 3), i * 30, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    node = nodes.RunMuxSaveVideo()
    frames, saved, count = node.save(video_url="", file_path=path)
    assert saved == path
    assert count == frames.shape[0] == 8
    assert frames.shape[-1] == 3

    # skip/cap/nth math applied at decode time
    frames2, _, count2 = node.save(
        video_url="", file_path=path, skip_first_frames=2, frame_load_cap=2, select_every_nth=2
    )
    assert count2 == 2
