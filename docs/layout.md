# Layout profiles and compact block summaries

Layout is a major floating panel in the alphabetically ordered PANELS pack.
All Blocks Collapsed and All Blocks Expanded are separate collapsible sections,
each with its own options and Minimize all/Expand all action.

Current control values are used immediately by the respective button. Save layout
preferences updates both profiles in the current `.vtw` working workspace as one
undoable edit; explicit Save workspace persists them to disk. Profiles are shared
by its architectures/tabs, not by all workspaces on the computer. Editing controls
alone does not store a preference; storing profiles does not move blocks. Defaults are:

| Profile | Style/direction | Layer gap | Block gap | Rearrange | Fit |
| --- | --- | --- | --- | --- | --- |
| Collapsed | Layered / LR | 48 | 16 | true | true |
| Expanded | Layered / LR | 80 | 32 | true | true |

Gaps are integer world/model units 8–1000, not pixels at the current zoom.
Direction LR places ranks left-to-right; TB places ranks top-to-bottom. Sockets
remain left/right in either mode. Disable Rearrange to preserve positions;
disable Fit to preserve the viewport. Each bulk operation, including fit when
possible, is one undoable edit. Both operations are disabled during training.
They leave computation, weights, data, bindings and preferred expanded sizes intact.

The persisted `metadata` key is `layout_preferences`; its TEXT value is JSON:

```json
{"version":1,
 "collapsed":{"style":"layered","direction":"LR","layer_gap":48,"block_gap":16,"rearrange":true,"fit":true},
 "expanded":{"style":"layered","direction":"LR","layer_gap":80,"block_gap":32,"rearrange":true,"fit":true}}
```

Only `layered` is implemented. Future styles need an explicit implementation and
validated option contract, not a selector entry pretending to work. Existing
workspaces lacking this optional key use independent defaults without being
rewritten on open. Invalid stored JSON/profiles are rejected. Root/profile keys
are closed; types are strict, including integer version and Boolean switches.
The old user configuration `layout` key is inert and is not imported into a
workspace; existing bytes are not rewritten merely to remove it. Appearance
(font/theme/spacing/zoom) remains user-level. The CLI `layout` edit and read-only
`inspect.layout_preferences` use the same pure contract in app/presentation.py.
Profile edits preserve computation, dataset bindings, results and immutable
scientific model revisions; no SQL table/container-version change is needed for
this optional presentation metadata. No-op profile updates create no undo entry.
Layout updates are refused without a workspace or during active training.

## Algorithm and evidence boundaries

Layer-based drawing is established in tools such as
[ELK Layered](https://eclipse.dev/elk/reference/algorithms/org-eclipse-elk-layered.html).
Its separation of direction, ordering and geometric spacing inspired this small
offline kernel. This is an independently implemented prototype, not an ELK port
and not an implementation of all its routing guarantees.

The kernel condenses strongly connected components, assigns longest-path ranks
to the resulting DAG, applies four alternating barycentric order sweeps with port
offsets, and centers each rank using actual effective block sizes. Adjacent ranks
and blocks respect configured clear gaps. Directed graph arrays, Concat source
order and router-port order remain untouched. Modules remain single blocks.
Cycles and disconnected drafts can be arranged without becoming valid models.
Overlarge coordinates or stale edges fail before replacing the original diagram.

No crossing-free or optimal edge-routing claim is made. Top-to-bottom layout does
not change socket placement. Future more elaborate styles may adopt a complete
port-constrained routing engine behind the same profile/action interface.

Node tests cover profile validation, deterministic/idempotent layout, spacing,
preferred sizes, ordering/semantic preservation, cycles, disconnected graphs,
many-port routers, atomic failures and 100 uneven 64-node DAG configurations.
Real Qt WebEngine tests exercise the buttons, single-entry undo, viewport restore,
workspace persistence, undo/redo control synchronization and unchanged scientific identity.

## Compact block content

Compacted blocks have a draggable type icon plus 9-world-unit monospace summaries:
input/output shapes, Linear last-axis in→out widths, activation names, reshape
shape information, or routing/reference summaries. Linear also shows Bias on/off
in an 8-unit line. Long text truncates to 13 characters with full hover detail;
unknown inferred widths show ?. Inspector retains the full contract.
Width is 72; height is at least 44, or 54 for Linear, and grows for multiple ports.
Single-click the compact icon selects it; drag moves it; double-click restores
details. Enter/Space toggles a focused icon. Folding changes presentation only.
