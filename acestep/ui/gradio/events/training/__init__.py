"""Training event-handler package.

This package intentionally avoids eager re-exports so Gradio startup can load
only the specific training submodules referenced by an active callback.
"""

