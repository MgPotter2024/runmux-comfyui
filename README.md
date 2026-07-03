# RunMux nodes for ComfyUI

Generate cloud video with **RunMux** (Seedance 2.0) right inside ComfyUI — no
local GPU required. These nodes call the RunMux cloud API through the official
[`runmux` Python SDK](https://pypi.org/project/runmux/). You drop a node, paste
your API key and a prompt, run, and get back a downloadable video URL (and,
optionally, the mp4 saved into ComfyUI's output folder).

## What you get

- **RunMux Generate Video** — text-to-video and image-to-video with up to
  **9 reference images**: wire ComfyUI "Load Image" nodes straight into
  `image_1..image_9`, and/or paste `asset://…` ids / image URLs into
  `reference_assets` (one per line; commas work too). `first_frame` /
  `last_frame` inputs give exact frame control. Outputs the video URL and,
  when downloading is enabled, the path to the saved mp4.
- **RunMux Save Video** — download the finished video into ComfyUI's output
  folder and decode it into an IMAGE frame batch (for preview, upscaling,
  interpolation, or chaining the last frame into the next shot).
- **RunMux Enroll Face** — register a (non-celebrity) face photo into your
  RunMux asset library. Wire an IMAGE straight in (or give a URL) and get back
  an `asset://…` reference for `reference_assets`.

All nodes live under the **RunMux** category in ComfyUI's node menu.

## Install

1. **Copy this folder** (or clone the repo) into your ComfyUI custom nodes
   directory:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/MgPotter2024/runmux-comfyui.git
   ```

2. **Install the RunMux Python SDK** into the *same* Python environment ComfyUI
   runs in:

   ```bash
   pip install runmux
   ```

   (If ComfyUI uses an embedded/portable Python, run the matching
   `python_embeded\python.exe -m pip install runmux` on Windows, or activate
   ComfyUI's venv first.) Upgrading from an older node pack? Also
   `pip install --upgrade runmux`.

3. **Restart ComfyUI.** The **RunMux** category appears in the node menu.

## Get an API key

Sign in at **https://console.runmux.com** (Google sign-in; first login creates
a workspace), open **API Keys**, and create one (looks like `sk-…`). Paste it
into the node's `api_key` field. Your key is your identity — don't share it.

## Using the nodes

### RunMux Generate Video

| Input | Notes |
|---|---|
| `api_key` | Your `sk-…` key. |
| `model` | `seedance-2-0-mini` / `seedance-2-0` / `seedance-2-0-fast`. |
| `prompt` | What to generate. Refer to wired images by order: 图1/Image 1 = `image_1`, … |
| `image_1..image_9` *(optional)* | Reference images, wired straight from "Load Image" nodes. Sent in slot order. |
| `reference_assets` *(optional)* | `asset://…` ids and/or public image URLs — one per line (commas ok). Merged after the wired images; 9 total max. |
| `image_url` *(optional)* | Legacy single-image field. Alone it keeps the old first-frame behavior; combined with references it becomes one more reference. |
| `first_frame` / `last_frame` *(optional)* | Exact start/end frame control. **Mutually exclusive with reference images** (upstream rule) — in reference mode, say "以图1为首帧 / use Image 1 as the start frame" in the prompt instead. |
| `resolution` | `480p` / `720p` / `1080p` (model-dependent). |
| `duration` | Seconds. |
| `auto_enroll_faces` *(optional)* | If on, raw face photos among the references are enrolled automatically before generation. |
| `download` *(optional)* | If on, the mp4 is saved into ComfyUI's `output/` folder. |

The node polls the job internally and returns once the video is ready, so you
get a finished URL — no manual waiting/looping. Outputs: **video_url** and
**file_path** (empty unless `download` is on).

Wired images are auto-encoded (JPEG, long side capped at 2048px) and uploaded
inline — you don't need to host them anywhere.

### RunMux Save Video

| Input | Notes |
|---|---|
| `video_url` | Wire from Generate Video's `video_url` output. |
| `file_path` *(optional)* | Skip re-downloading if Generate Video already saved the mp4. |
| `save_subfolder` / `filename_prefix` | Where/how to save under `output/`. |
| `frame_load_cap` | Max frames to return (0 = all). |
| `skip_first_frames` / `select_every_nth` | Frame sampling controls. |

Outputs: **frames** (IMAGE batch), **file_path**, **frame_count**. Decoding
uses PyAV (bundled with current ComfyUI) or OpenCV as a fallback.

### RunMux Enroll Face

Wire a portrait IMAGE in (or give a public photo URL) of a **non-celebrity**
person; it returns an `asset://…` reference. Put that into Generate Video's
`reference_assets` for a faithful likeness. Celebrity / copyrighted faces are
rejected by moderation (you'll get a clear reason).

## Notes on faces

- Use ordinary people's photos; celebrity or copyrighted likenesses are refused.
- To lock a specific person, enroll them once and reuse the `asset://…`
  reference — more reliable than re-uploading raw photos each run.
- Failures raise a clear error explaining why.

## Links

- RunMux console: https://console.runmux.com
- Python SDK: https://pypi.org/project/runmux/
- API base: https://api.runmux.com

## License

MIT
