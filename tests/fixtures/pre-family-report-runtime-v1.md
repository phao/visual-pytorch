# Immutable pre-family report runtime fixture

`pre-family-report-runtime-v1.zip` is a Git archive from product commit
`bc85e57254083bf969a189d9685574b4d8598669`, before family/specialization storage.
SHA-256: `d169eccc0681f13750e22479f6e3192d4a9f5f16abdc6344110984e5224764a0`.
It contains that commit's 21-file numerical runtime, its `app/notebooks.py`
producer location and GPL-3.0-only LICENSE. The producer file supplies only a
source location to the current export test; it is not an extra runtime file.
No observations, checkpoints, private records or credentials are present.

The compatibility test creates a version-1 saved-result report bundle from these
immutable bytes, verifies it with the current verifier, and executes its notebook
in a fresh real kernel. Original inputs remain unchanged. Do not replace this
fixture with current source to make compatibility pass.
