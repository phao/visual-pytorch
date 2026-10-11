# Immutable pre-Softmax report runtime fixture

`legacy-report-runtime-v1.zip` is a Git archive of the exact GPL-3.0-only runtime
closure and LICENSE from product commit
`5a62d89f7b0c976a8b7a95e8f49800c5587edd8a`, before the activation-module extraction.
Its SHA-256 is `7114ff088dd8acfb72bc92704cadaf45473ee1b0b61c222a4d62615a269e1c8a`.
It contains the 20 files in that commit's `app.notebooks.RUNTIME` and LICENSE;
there are no observations, checkpoints, private campaign records or credentials.

This is an immutable compatibility test fixture, not another live product copy.
The test constructs a version-1 report bundle from this copied closure and runs
its requested notebook in a fresh kernel using the current bundle verifier.
This verifies that adding a source dependency does not revoke older bundles.
Do not regenerate this fixture from current source to make compatibility pass.
