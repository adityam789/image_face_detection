"""
facesorter.search
Natural-language photo search: structured filters (date, place, people) plus embedding search.

Heavy dependencies (PyTorch, onnxruntime, MLX, Gradio) are imported lazily inside the
modules that need them, so ``import facesorter.search`` stays cheap. See
``docs/search-design.md`` for the architecture.
"""
