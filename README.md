# RunMux nodes for ComfyUI

Generate cloud video with **RunMux** (Seedance 2.0) right inside ComfyUI — no
local GPU required. These nodes call the RunMux cloud API through the official
[`runmux` Python SDK](https://pypi.org/project/runmux/). You drop a node, paste
your API key and a prompt, run, and get back a downloadable video URL (and,
optionally, the mp4 saved into ComfyUI's output folder).

## What you get

- **RunMux Generate Video** — text-to-video and image-to-video. Outputs the
  video URL and, when downloading is enabled, the path to the saved mp4.
- **RunMux Enroll Face** — register a (non-celebrity) face photo into your
  RunMux asset library and get back an `asset://…` reference to use as the
  `image_url` of the Generate Video node.

Both nodes live under the **RunMux** category in ComfyUI's node menu.

## Install

1. **Clone this repo** into your ComfyUI custom nodes directory:

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
   ComfyUI's venv first.)

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
| `model` | e.g. `seedance-2-0-mini`. |
| `prompt` | What to generate. |
| `image_url` *(optional)* | A public image URL **or** an `asset://…` from Enroll Face, used as the first frame / reference. |
| `resolution` | `480p` / `720p` (model-dependent). |
| `duration` | Seconds. |
| `auto_enroll_faces` *(optional)* | If on, a face in `image_url` is enrolled automatically before generation. |
| `download` *(optional)* | If on, the mp4 is saved into ComfyUI's `output/` folder. |

The node polls the job internally and returns once the video is ready, so you
get a finished URL — no manual waiting/looping. Outputs: **video_url** and
**file_path** (empty unless `download` is on).

### RunMux Enroll Face

Give it a public photo URL of a **non-celebrity** person; it returns an
`asset://…` reference. Feed that into Generate Video's `image_url` for a
faithful likeness. Celebrity / copyrighted faces are rejected by moderation
(you'll get a clear reason).

## Notes on faces

- Use ordinary people's photos; celebrity or copyrighted likenesses are refused.
- To lock a specific person, passing their photo as `image_url` is more reliable
  than scattering it across reference images.
- Failures raise a clear error explaining why.

## Links

- RunMux console: https://console.runmux.com
- Python SDK: https://pypi.org/project/runmux/
- API base: https://api.runmux.com

## License

MIT
