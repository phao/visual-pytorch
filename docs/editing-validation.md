# Selection and destination-edit validation

The successor uses existing .vtw node/edge records: no SQLite schema migration,
new source-port identity, or changed CLI command grammar is required for these
gestures. The portable UI behavior is specified in UI-AGENT-HANDOFF.md; the CLI
equivalent and intentional explicit-batch differences are in vpw-cli.md.

## Checks

- Graph helper tests cover atomic invalid/occupied/cyclic/stale drops, unchanged
  drops, router source-port identity, unaffected sibling fanout, ordered Concat
  removal/append and duplicate feature prevention.
- Real sandboxed Qt WebEngine smoke drives destination dragging and D through
  native mouse/key events, verifies persistence only on release, undo/redo,
  blank/Escape cancellation, typing/training guards and old wire double-click.
- CLI regression edits a private .vtw in one batch, reopens/validates/exports
  the new path, and confirms stale replay cannot change file bytes.
- Existing Python/JS/native Qt suites continue to cover MLP execution, workspace,
  source-port/export, native menus, display preferences and theme behavior.

## Test design reference

Qt's [QTest mouse implementation](https://raw.githubusercontent.com/qt/qtbase/6.4/src/testlib/qtestmouse.h)
deliberately separates ordinary click timestamps. The QWindow double-click
overload emits paired native events together; the smoke therefore uses that
overload rather than treating two QWidget mouseClick calls as a browser
double-click. This choice is checked against the actual installed WebEngine,
with no disabled Chromium sandbox or synthetic DOM double-click in that test.

Desktop identity checks confirm the installed entry points to this campaign.
Real user-session finder/Alt-Tab appearance and GPU rendering remain acceptance
checks, not claims made by these offscreen tests. The installer still reports
its existing generated-icon-cache warning; the installed SVG bytes are unchanged.
