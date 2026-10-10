# Install the public prototype

The initial release is source-only. Use Python 3.10 or newer, a graphical desktop
and an environment containing PyQt6, Qt WebEngine and PyTorch. Linux is the
currently exercised platform; Windows/macOS installation is not yet verified.
Requirements specify supported broad ranges, not a lockfile or a promise that
every Python/OS/dependency combination has been tested.

```sh
git clone https://github.com/phao/visual-pytorch.git
cd visual-pytorch
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

This installs dependencies into the local environment. No account, server,
network access for the editor or GitHub credential is required to use the app.
Package download sizes, particularly PyTorch, can be substantial. For a particular
CPU/GPU wheel, follow the [official PyTorch installer](https://pytorch.org/get-started/locally/)
and then install the Qt requirements; this prototype's training path uses CPU.

## Optional Linux desktop integration

```sh
python install_desktop.py --dry-run
python install_desktop.py
```

This per-user installer registers the application/icon and `.vtw` Open With
candidate. It does not force a file association or install system packages.
If you move the checkout or recreate its Python environment, rerun the installer
from the new location. Application-finder/Alt-Tab behavior depends on the desktop
session; inspect dry-run output first.

## First launch and troubleshooting

- Use an activated environment. Verify `python -m pip --version` points there.
- Qt may need operating-system X11/Wayland/OpenGL libraries. Read the terminal's
  actual missing-library/plugin error; system package names vary by distribution.
- Start in a normal graphical user session, not as root. Do not disable Chromium
  sandboxing to work around an installation failure.
- For a graphics/compositing problem, try `python run.py --renderer software`;
  the launcher also has a scoped auto workaround. Inspect `python run.py --help`
  for the actual available flags. This is not a guarantee for every graphics stack.
- Separate environments are supported: `python -m app.desktop --torch-python /absolute/path/to/python`.
  The selected worker interpreter must import PyTorch.
- Appearance/RAM preferences currently use the compatibility path
  `~/.tensorloon/config.json`. Workspace layout profiles are in `.vtw`, not that
  global file. Save workspace explicitly to persist changes.

Report failures with OS, Python/Qt versions, the command and a redacted error
message. Never include credentials, private datasets or sensitive workspaces.
