"""Process-local WebEngine rendering policy; no Qt imports or system changes."""
import os
import sys


def configure_renderer(mode="auto", webengine_version="", environ=None, platform=None):
    """Call before QApplication/WebEngine contexts are constructed.

    Auto mitigates the user's observed Qt WebEngine 6.9.2/Wayland dropdown flip.
    This narrow guard is not a statement that every other version is unaffected.
    Existing Chromium flags are preserved, including explicit user choices.
    """
    if mode not in {"auto", "software", "native"}:
        raise ValueError("Renderer must be auto, software or native.")
    env = os.environ if environ is None else environ
    system = sys.platform if platform is None else platform
    qpa = env.get("QT_QPA_PLATFORM", "").split(";")[0].split(":")[0]
    wayland = qpa.startswith("wayland") if qpa else (
        env.get("XDG_SESSION_TYPE") == "wayland" or bool(env.get("WAYLAND_DISPLAY")))
    observed_config = system.startswith("linux") and wayland and webengine_version == "6.9.2"
    flags = env.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    already_software = any(token.split("=", 1)[0] == "--disable-gpu" for token in flags.split())
    reason = "Qt default rendering"
    if mode == "software":
        reason = "Explicit software rendering request"
    elif mode == "auto" and observed_config:
        reason = "Qt WebEngine 6.9.2/Wayland dropdown rendering workaround"
    software = already_software or mode == "software" or (mode == "auto" and observed_config)
    if software and not already_software:
        env["QTWEBENGINE_CHROMIUM_FLAGS"] = flags + (" " if flags else "") + "--disable-gpu"
    elif already_software:
        reason = "Software rendering requested by existing Chromium flags"
    return {"mode": "software" if software else "native", "reason": reason,
            "webengine_version": webengine_version}
