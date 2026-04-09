"""
Read / validate / write the Codebase settings.json file.

Flask uses this to expose the running pipeline's configuration to admins on
the dashboard.  The validation rules here MUST stay in sync with
``Codebase/main.py::validate_settings`` — that file is the source of truth for
what the runtime considers a legal settings.json.
"""

import json
import os
import tempfile
from pathlib import Path


class SettingsError(ValueError):
    """Raised when settings.json contents fail validation."""


def load(path: Path) -> dict:
    """Read and parse settings.json.  Raises SettingsError on parse failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SettingsError(f"settings.json not found at {path}")
    except json.JSONDecodeError as e:
        raise SettingsError(f"settings.json is not valid JSON: {e}")


def load_text(path: Path) -> str:
    """Read settings.json verbatim (for displaying in the editor)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise SettingsError(f"settings.json not found at {path}")


def parse(text: str) -> dict:
    """Parse a JSON string into a settings dict, raising SettingsError."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SettingsError(f"Invalid JSON: line {e.lineno} col {e.colno}: {e.msg}")


def validate(settings: dict) -> None:
    """Validate a parsed settings dict.

    Mirrors the rules in ``Codebase/main.py::validate_settings``.  Keep these
    two implementations aligned — if you add a rule there, add it here.
    """
    if not isinstance(settings, dict):
        raise SettingsError("Top-level settings must be a JSON object")

    for key in ("default_mode", "modes", "cameras"):
        if key not in settings:
            raise SettingsError(f"Missing required key: '{key}'")

    if not isinstance(settings["modes"], dict):
        raise SettingsError("'modes' must be a JSON object")

    if settings["default_mode"] not in settings["modes"]:
        raise SettingsError(
            f"default_mode '{settings['default_mode']}' is not present in 'modes'"
        )

    if not isinstance(settings["cameras"], list):
        raise SettingsError("'cameras' must be a JSON array")

    for cam in settings["cameras"]:
        if not isinstance(cam, dict):
            raise SettingsError("Every entry in 'cameras' must be an object")
        cam_id = cam.get("id", "unknown")

        enabled = cam.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise SettingsError(
                f"Camera id:{cam_id} 'enabled' field must be true or false"
            )

        if not cam.get("url"):
            raise SettingsError(f"Camera id:{cam_id} is missing a 'url' field")


def save_atomic(path: Path, settings: dict) -> None:
    """Write settings.json atomically.

    Writes to a sibling tempfile and ``os.replace``s it onto the target so a
    crash mid-write cannot leave a half-written settings.json on disk.  This
    is atomic on both POSIX and Windows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".settings.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
