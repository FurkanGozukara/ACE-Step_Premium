"""Media path reference repair for Auto-Editor workflow exports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def rewrite_fcpxml_media_references(
    workflow_path: str | Path,
    media_path: str | Path | None,
) -> bool:
    """Rewrite FCPXML original-media references to an absolute file URI.

    Args:
        workflow_path: FCPXML file to update in place.
        media_path: Media file that the editor workflow should reference.

    Returns:
        ``True`` when at least one media reference was updated.
    """

    if not media_path:
        return False

    workflow = Path(workflow_path)
    if workflow.suffix.lower() != ".fcpxml" or not workflow.is_file():
        return False

    media_uri = _file_uri(media_path)
    tree = ET.parse(workflow)
    changed = _rewrite_original_media_reps(tree.getroot(), media_uri)
    if changed:
        tree.write(workflow, encoding="utf-8", xml_declaration=True)
    return changed


def _rewrite_original_media_reps(root: ET.Element, media_uri: str) -> bool:
    """Update original-media ``media-rep`` elements and report whether they changed."""

    changed = False
    for element in root.iter():
        if _local_name(element.tag) != "media-rep":
            continue
        if element.get("kind", "original-media") != "original-media":
            continue
        if element.get("src") == media_uri:
            continue
        element.set("src", media_uri)
        changed = True
    return changed


def _file_uri(path: str | Path) -> str:
    """Return an absolute ``file://`` URI for a local media path."""

    return Path(path).expanduser().resolve().as_uri()


def _local_name(tag: str) -> str:
    """Return an XML tag name without a namespace prefix."""

    return tag.rsplit("}", 1)[-1]
