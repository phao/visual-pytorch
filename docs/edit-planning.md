# Inspectable construction and edit previews

`template-list` and `template-show NAME` describe five pure construction recipes
without opening a workspace, reading user settings, importing PyTorch/Qt or
training. [Recipe declarations](construction-recipes.md) distinguish exact
conditional models, parameter components and approximate estimators. They never
create data, latent labels or implicit fits.

Use the normal workspace CLI to preview or admit a recipe:

```sh
python vpw_cli.py template-list
python vpw_cli.py template-show neural-msvar
python vpw_cli.py study.vtw instantiate-template neural-msvar --id msvar --dry-run
python vpw_cli.py study.vtw instantiate-template neural-msvar --id msvar --options options.json
```

`options.json` is a closed typed JSON object; for example
`{"K":5,"p":3,"D":3,"T":12,"hidden":32,"controlled":true}`. The resulting
ordinary diagrams are completely editable. References are created before their
consumers; collisions refuse instead of replacing existing work.

## Before committing a patch

`plan-edits PATCH.json` and `apply PATCH.json --dry-run` execute the same ordered
edit service against a private in-memory SQLite clone. The original database,
disk bytes, dirty flag, browsing selections, undo/redo history, random state and
configuration are unchanged. The default clone/declaration budget is64MiB and
at most256 edits; this is a preview resource guard, not a neural dimension limit.

The JSON plan includes changed architectures/nodes/fields, before/after stable
port and parameter-owner contracts, dataset bindings, result invalidation,
reference dependencies, selected immutable heads and workspace metadata changes.
Tensor packing remains contract order, never alphabetical order. Failures have
stable codes, edit/architecture paths, messages and repair guidance. Partial
effects from a failed preview are marked and cannot be admitted.

Whole-workspace semantic validation is the default. Existing invalid drafts are
reported separately from errors introduced by the proposed patch. `--allow-invalid`
permits structurally valid drafts, reports status `draft`, and does not establish
executability or silence validation findings. Scientific immutable objects still
use their normal exact validators.

Actual admission reruns the preview checks and publishes the entire checked
working database at once. `--expect-snapshot SHA256` additionally requires the
exact prior in-memory preview snapshot. For concurrent file protection, always
use the independently reported disk revision:

```sh
python vpw_cli.py study.vtw --expect-revision FILE_SHA256 apply patch.json --expect-snapshot PREVIEW_SHA256
```

The file save still performs its existing external conflict check. A successful
preview does not authorize a later overwrite after another user/agent edits the
file. A late failed edit does not publish partial changes or save the workspace.

## Desktop bridge

The native Qt bridge uses `templateList`, `templateShow`, `previewTemplate`,
`instantiateTemplate`, `previewEdits` and `applyEdits` with the same services.
`previewTemplate`/`instantiateTemplate` require `name` and the explicit `id`;
optional `options` holds the same typed JSON object. Admission optionally takes
`expected_snapshot`. Edit requests require `edits`, optionally `allow_invalid`.
Only actual admissions are history mutations, with one undo step for the whole
batch. No-op admission adds none. Edits are blocked while a run is active.

These bridge operations do not add a new menu/workbench; the current diagram UI
and floating-panel behavior remain unchanged. Existing ordinary construction and
family authoring continue to work. There is no production loss-panel integration.

## Application doctor

`python vpw_cli.py study.vtw doctor` uses the normal application loader, then
checks SQLite integrity/references and current graph contracts. Invalid drafts
get located findings and a nonzero exit status. Unsupported/corrupt files are
refused by the same loader as normal opening; no alternate database parser is
introduced. The file, history, configuration and selections remain unchanged.

The environment section reads package metadata in the CLI interpreter without
importing PyTorch/Qt or running kernels. Package presence/version is not proof
that numerical imports, notebooks or GPU execution work. This command is separate
from campaign/CPW workspace diagnostics.
