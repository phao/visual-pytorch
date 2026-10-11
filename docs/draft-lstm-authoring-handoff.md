# LSTM authoring handoff

## Resumed implementation — 2026-10-11

Creation/editing forms, the strict shared CLI input seam and dedicated pure/
serialized/numerical/mounted-form/native checks now exist on this child branch.
See [authoring specification](lstm-authoring.md). Tests exposed and repaired a
real default-prefill no-op mutation; exact SQL/history now remain unchanged.
Preserve the historical draft failures below as evidence, not current defects.
Full1132 Python tests pass (3 Qt-only skips separately covered), as do352 JS and
actual sandboxed forms. This bounded authoring result is ready for integration
into retained D only; actual commit/integration evidence is in the private
campaign handoff. The historical checkpoint below is not the current tip.

The first full1132-case run had one obsolete broad UI cap assertion: LSTM's
64-layer grammar was mistaken for an old constrained-run limit. The test now
allows only that explicitly identified stack-depth field, retaining refusal of
caps on other size controls. The original failure is kept. Actual form captures
also exposed retained dialog scroll hiding the reopened title; open resets it.
Current tests cover real native Apply/Cancel/no-op/refusal/stale/history/reload,
all authored/native/exported input/parameter gradients against an independent
gate scan and35-update fitting. No user workspace/configuration is used.

Remaining: explicit new LSTM family/runtime-T grammars and original D acceptance,
then E. LoRA/MS-VAR report requests remain deferred; no policy/init/likelihood
semantics have been smuggled into the existing operator. Public A–C is unchanged.

## Historical initial incomplete checkpoint

Status: draft, not accepted or integrated. Branch `task/stage-d-lstm-authoring`
starts at `720d7db0fd7f0393e1fb7781382c723d4d50455e` (retained local D).
The public A–C branch is not changed by this checkpoint.

## Preserved work

`app/lstm_authoring.py` drafts allocation-free branch creation and in-place
constructor/interface edits with exact graph/reference tokens. Creation exposes
sequence, initial hidden/cell states and three returned ports. Edits preserve
physical IDs, existing outgoing wires and unrelated drafts; explicit compatible
source bindings are required instead of silently resizing state Inputs.
Desktop bridge and CLI prepare/apply routes call the same builder. Parameter
layout counting and affected-path proof include LSTM.

No palette creation button, JavaScript form, Inspector editor, UI authoring tests,
or dedicated builder/CLI regression tests have been added. The existing imported
LSTM diagram remains the separately verified UI capability. Do not describe this
checkpoint as finished LSTM authoring or as D completion.

## Checks and unresolved issue

On 2026-10-11, system Python ran these existing modules:

```
PYTHONPATH=.:tests /usr/bin/python3 -m unittest \
  test_lstm_contract test_recurrent_authoring \
  test_recurrent_reconfiguration test_node_presentation \
  test_task_context test_cli_config -v
```

76 tests ran in 0.660 s: 72 passed and four failed. The failures are the CLI
resize, move/conflict, destination reconnection and invalid-draft tests. Those
tests mock `vpw_cli.read_json`, while the new strict `apply` parser reads input
directly. Consequently their mocked batches are bypassed and stdin parsing
fails. Do not weaken JSON admission or silently revise tests to hide this issue;
first define the intended input seam and verify real serialized CLI requests,
duplicate-key/nonfinite refusal and unchanged-file failure behavior. Database
ResourceWarnings were also present. No numerical, full-suite or actual native
desktop authoring acceptance has been run for these draft changes.

## Next step

Resume this child branch, repair/verify strict CLI admission and add independent
builder/serialized CLI tests. Then implement typed creation and stable-interface
editing forms, with stale/reload guards, error-draft retention and atomic history.
Run numerical and real sandboxed Qt checks before merging into retained D.
Continue new explicit family/runtime-T LSTM grammars only after authoring
acceptance. Preserve old grammar meanings and all independent scientific evidence.

The draft was checkpointed while new user-supplied feature reports were being
assessed, not because those requests were implemented. Return the everyday
checkout to the previously verified retained D branch until this draft resumes.
