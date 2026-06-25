"""RunMux custom nodes for ComfyUI.

Generate cloud video with RunMux (Seedance 2.0) and enroll faces into the asset
library — all through the official ``runmux`` Python SDK. No local GPU required.

ComfyUI discovers a custom-node pack by importing this package and reading
``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS`` from it.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
