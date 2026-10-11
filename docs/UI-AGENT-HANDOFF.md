# Reusable floating-canvas GUI: handoff for other projects

Audience: the owner and agents adapting this design to **other software projects**.
This is not the current campaign's continuation plan. It is a reusable interaction
specification, grounded in implemented code, with explicit boundaries between
general GUI mechanics and Visual PyTorch's domain-specific neural-network behavior.

## 1. Product idea and invariants

The canvas is the main workspace, not the space left over after permanently open
sidebars. Dense, compact boxes and directed wires communicate structure at close
range and network scale when zoomed out. Tools form a separate floating layer.
A shared vertical launcher pack gives access to independently movable rectangular
menus. All menus start hidden; the pack starts expanded into its button list.
The launcher starts at stage (0,0), flush to the canvas's top-left, below the
application header. Reset restores this CSS origin. It remains freely movable;
this is an initial position, not permanent docking or an OS-screen coordinate.

Preserve these invariants in a reuse:

1. Opening a menu does not reduce canvas geometry or shift the diagram.
2. Closing/hiding a menu does not delete its contents, selections or edited data.
3. Canvas coordinates and menu coordinates are independent; translating menus
   never changes model/world positions, graph zoom or wires.
4. The shared pack can reduce **all tools to one floating icon**.
5. Menus can overlap and leave the window; a recoverable slice remains visible.
6. Standard controls, text selection, scrolling and modal dialogs stay usable.
7. A box/wire means a real domain operation/relation, not just decorative geometry.
8. The GUI must show invalid drafts clearly, without confusing them with run policy.

Future menu or box/link ideas should extend this explicit contract rather than
being treated as already implemented features. This document is a baseline,
not a prohibition on improving the design.

Domain-specific update: a concrete LSTM drawing seeds Family7 with a null runtime
policy in the existing owned architecture-family editor. Typed cards/staged JSON
support checked integer selection expressions; preview/retention/materialization
remain separate. This does not change general floating-panel mechanics or imply
runtime-T LSTM support. See [LSTM family contract](lstm-families.md).

## 2. Coordinate systems and stacking

| Layer | Coordinates | Persisted in Visual PyTorch | Movement |
| --- | --- | --- | --- |
| Canvas world | Model/world units | Box positions/sizes; camera session-only | Block drag; Ctrl inverts block snap default; empty drag pans |
| Floating menus | CSS viewport units relative to editing stage | Explicit workspace preset; ordinary moves tab-local | Menu drag; Shift-drag translates menu layer |
| Whole interface | Qt WebEngine zoom | App config JSON | Ctrl+/Ctrl−; not canvas zoom |

Canvas transforms are translate(view.x,view.y) followed by scale(view.zoom).
Convert pointer positions through the inverse transform before moving a box.
Snap grid is 24 world units for boxes, 24 CSS stage units for menus. Grid origin
is (0,0) in each layer, not the window decoration's top-left. UI zoom scales page
appearance, but does not replace these logical coordinate definitions.

The single application header wraps its title/tools and workspace actions without
overlapping groups. At extreme interface zoom it is a focus-scrollable region,
capped at 40vh, rather than pushing controls outside the window or consuming the
canvas. The footer wraps, with a 15vh scroll cap. The launcher pack is capped to
the stage; its list can scroll independently of its stationary grip. Ordinary
floating-panel content still uses its existing overlay rails. These are overflow
fallbacks, not hidden duplicate commands or reductions to the user's font size.
Full truncated architecture/file names remain available as hover titles. Native
single/multiple-tab checks cover real control focus/hit tests and mouse clicks.

The editing stage is positioned, isolated, clipped to its rectangle. Menus are
absolute descendants, not OS top-level windows. They therefore cannot draw over
window chrome or outside the application's actual surface. "Off-window" means
most of the rectangle can be outside that clipped stage; recovery uses its slice
that is still visible. Native detached palette windows are NOT implemented.

All floating menus, including the PANELS launcher pack, start at z-index 2,
above the graph. Pointer interaction or opening raises the target through one
shared monotonic counter. The pack is not always on top: a clicked/opened palette
can cover it, and clicking the exposed pack brings it back above other palettes.
Expansion raises a palette but does not reserve a permanently higher tier;
restoring ordinary geometry preserves its current stacking level. Pack collapse
and keyboard/double-click restoration raise the pack. Reset floating menus is
the recovery command when it is completely covered. Popups/modal dialogs use
their browser/native mechanisms; notifications remain above the menu layer.

## 3. Launcher pack state machine

Pack states: expanded button list or collapsed icon. Menu visibility is independent
of pack visibility; ordinary usage is any subset of menus open.

| Action | Result |
| --- | --- |
| Pack button for hidden menu | Open and raise that menu; update aria-expanded |
| Pack button for visible menu | Hide it; retain contents and geometry |
| Click pack icon while pack expanded | Hide all menus, restore any full-stage expansions, collapse pack to one icon |
| Single click collapsed icon | No restoration; permits moving it without accidental expansion |
| Double-click collapsed icon | Restore button list only; menus remain hidden |
| Enter/Space on focused collapsed icon | Same restoration, with preventDefault |
| Reset panels command | Reset geometry, clear full-stage expansion, hide all menus, restore expanded pack |

The collapsed icon remains the event/click target during drag so browser
double-click recognition survives. Dragging it must not leave fixed 42px dimensions
on the expanded pack. Clear compact width/height before restoring natural size.

Current pack menus are Architectures, Building blocks, Experiment, Inspector and Layout,
in alphabetical browsing order.
Only their labels/content are domain-specific. There is no architectural reason
to limit other projects to five menus. Each menu must have exactly one launcher
binding. Menus are rectangles: they can be wide/horizontal, not mandatory sidebars.

## 4. Menu state, controls and geometry

Each menu has hidden/visible state and, while visible, ordinary/expanded state.
Expanded means inset 16 CSS units from the stage edges, not native fullscreen.
Store the previous inline style before expanding; restore it before hiding or
returning to ordinary state. Reset removes stored restoration state too.

Controls:

- × hides the menu and returns focus to its launcher button.
- ⤢ expands; its restored state uses ↙ and aria-pressed to communicate toggling.
- Bottom-right grip resizes; focus + arrow keys changes width/height by 20 units.
- A title/drag handle and exposed frame allow movement.
- Empty content areas also allow movement; controls/text do not.

Pointer/keyboard minima use computed typography: common width max(220,17f),
Inspector/Architectures width max(220,20f), height max(140,10f), where f is base
font size in CSS pixels. Expanded panels are exempt. The launcher is intrinsic
max-content width with nowrap buttons, natural height and minimum width 150;
dragging never freezes its expanded dimensions. A ResizeObserver reconstrains it
when font/content changes. Collapsed geometry remains 42 × 42.
Ordinary palettes can be larger than the stage. Let the rectangle leave bounds,
but constrain x to [24−width, stage.width−24] and y to
[24−height, stage.height−24]. This guarantees an intersecting 24-unit slice on each
axis, potentially a corner. Frame padding (currently 5 units) is draggable even
when the title is offstage. It does not guarantee the whole title remains visible.

Grid rules:

- Independent user Display preferences `grid_snap` (blocks) and `menu_grid_snap`
  (panels/pack/layer) default false; Ctrl inverts the appropriate one per event.
  Preview/Cancel/Save behave like other display preferences. Shift retains its
  menu-layer role; group selection remains deferred.
- Snapped menu/pack drag: align its **top-left** to the nearest 24-unit grid point.
- Snapped resize: align **bottom-right**, preserving top-left where feasible.
  Choose a grid-aligned corner satisfying minimum size.
- Shift layer-drag with snapping active: snap pack top-left as the shared anchor; preserve
  relative positions by applying the same delta to ordinary menus, including hidden ones.
- Visibility constraints can override exact snapping at stage extremes. Expanded
  menus do not move/resize and are excluded from layer translation until restored.

On stage resize, refit expanded menus and constrain moved ordinary rectangles/pack.
Closed menus need no immediate painted layout; when measuring one, briefly remove
hidden synchronously and restore it before the next paint. Never interpret a
display:none zero rectangle as its stored geometry.

### Explicit workspace panel preset

Layout → Floating panel arrangement saves the current ordinary rectangles,
visibility, expansion flags, pack location/collapse and bottom-to-top order to
`metadata.panel_layout`. Save workspace commits the draft to `.vtw`. New tabs and
workspace opens initialize from it; Restore saved arrangement reapplies it.
Ordinary gestures remain tab-local and create no history steps. Other open tabs
receive updated preset metadata without immediately moving. Reset is local.
Opening from the Architectures list clears selection/hides all panels/collapses
the pack after success, including tab-hint/Ctrl/Meta/double-click opens. A new tab
suppresses preset visibility once at initial load; ordinary navigation is unchanged.
Failed opens do not hide the layer. The saved preset is not rewritten.
Font/viewport adaptation keeps readable minima and a 24-unit recoverable slice,
without rewriting the saved geometry. Capture expanded panels' ordinary restore
rectangle, not full-stage bounds. Presets are outside computational undo/redo;
restoring a historical graph must preserve the current preset.

The precise closed JSON schema, CLI operation, stale-write guards and global
snap-default distinction are in [panel-workspace.md](panel-workspace.md). Treat
these as app storage bindings: a different product may use its own equivalent.

## 5. Gesture priority and event ownership

This priority prevents the same drag from moving both a menu and a box:

1. Native/page-modal dialog handling wins while modal.
2. Stage **capture-phase** Shift + primary-button pointerdown starts menu-layer
   translation, except inside input/select/textarea. Capture the pointer on stage,
   preventDefault and stopPropagation. Expanded menus are excluded.
3. Resize grip starts resize; Ctrl modifies its corner snapping, not movement.
4. Title/pack grip or eligible empty menu area starts menu/pack movement.
5. Canvas socket wiring, box drag or empty-canvas pan handles ordinary events.

Use pointer capture and finish on pointerup, pointercancel and lostpointercapture.
Also finish on window blur. Do not leave a stale active drag after a canceled gesture. Only primary button
starts these drags. Native context menus are separate, normally right-click.
During active gestures, clear text selection and temporarily suppress user-select
through root loom-dragging/loom-resizing classes. Movement shows grabbing;
resize shows nwse-resize. Idle diagram text is never selectable; idle panel text
still is. Cleanup must remove cursor/selection classes even on cancel/blur.

Empty-area drag eligibility in current code:

- The panel frame itself is eligible.
- Descendants of a title handle are handled by that handle, not a second drag.
- button/input/select/textarea/a/label/summary/pre/code/contenteditable,
  data-no-drag regions, scrollbars and resize grips are excluded.
- Other text remains selectable: walk descendant text nodes and inspect their
  range rectangles against pointer coordinates. Only whitespace outside actual
  text rectangles is draggable. Do not reject an entire content section merely
  because it has some text elsewhere.
- The test DOM has no text-range layout; its fallback excludes nonempty text.
  That fallback is a test limitation, not a browser geometry proof.

Shift is a stronger layer gesture, so Shift-drag on a title or most menu buttons
translates the layer; input/select/textarea editing remains excluded. Use this
deliberately when adapting to a domain where Shift already selects canvas objects.

## 6. Scrolling and space-efficient menus

Each menu has one primary scrollport with overflow:auto and native vertical
scrollbar hidden. Its overlay rail is outside scrollport content, absolutely
positioned on the right (4-unit offset), between title and resize corner.
Thumb width is 6 units; hover/focus can enlarge it to 8. Its minimum length is 26.
Scroll range and thumb travel are derived from viewport/content/rail height.

Show rail on panel hover, focus-within or recent scrolling; fade after 850ms.
Reveal rail BEFORE measuring it: a hidden element has zero clientHeight.
ResizeObserver and content MutationObserver update geometry. Keep the rail outside
the observed subtree to avoid a mutation-observer feedback loop.

Interactions: thumb drag, track click/drag, and focused thumb ArrowUp/Down (40
units), PageUp/Down (80% viewport), Home/End. Set role=scrollbar, vertical
orientation, aria-controls and min/max/current values. Wheel/native scrollport
behavior remains normal; menu drag must not consume scrollbar gestures.

Avoid nested scrollbars for normal menu sections. Architectures has its own
major menu, a non-collapsible Currently open bookkeeping card and then its list.
The title bar owns the add button; there is no redundant Workspace models heading.
The card shows escaped/plain name, graph counts and valid contract stats, or
checking/draft status; it is not another scrollport. The list has natural
height, max-height:none, overflow:visible; only the outer menu scrolls if needed.
Building blocks still contains a domain-specific selector for reusable architectures.
Browsing the collection and selecting a reusable block are different tasks.
Every block button shows only a type icon and name. Help lives in its native
title tooltip and aria-label, not permanent secondary text. Reuse this compact
pattern but retain explanatory content, especially in didactical software.

When a selector configures a specific add/action button, put its label, selector
and button inside one named semantic group (`role="group"`, `aria-labelledby`).
Visual PyTorch uses `.palette-group` for Activation and Architecture reference.
A one-pixel `--line` border, 7px corners and one menu-spacing unit of internal
padding/gap hint at their relationship without making another heavyweight panel.
Child button/label margins are zero inside the group. Keep the group intact when
sorting browsing entries and retain selector-before-action keyboard order.
Standalone buttons are not framed; grouping never changes their actions, graph
or persistence. Reuse the pattern for genuinely coupled controls, not every row.

Code blocks may have independent horizontal overflow. Do not claim that every
possible nested scrollbar is forbidden; the actual rule concerns normal panels.

## 7. Canvas boxes, links and inspection

General reusable concepts:

- SVG world group containing wires behind boxes; dotted grid anchored to world transform.
- Compact rounded boxes; stable IDs separate from displayed titles.
- Shape/type/detail summaries; ellipsis for long labels with full hover text.
- Ports owned by the box contract, not inferred from its visual position.
- Directed cubic Bézier wires with endpoints anchored to port coordinates.
- Selection border and distinct port hover/armed state.
- Double-click any box opens Inspector for that exact box.
- Wire removal by double-click targets the exact edge, including source-port identity.
- Fit chooses the largest axis-aligned canvas rectangle not covered by visible menus/pack.
  Insets are 50 CSS stage units per side on roomy rectangles, decreasing to 15%
  of each dimension on narrow ones. Every positive region is usable, including
  strips below 80 units; absent/zero space requests hiding/moving a menu. Fit
  never forces a minimum scale that would overflow the available rectangle.

Box compactness is separate from menu compactness. User setting box_spacing is
an integer 1–100, default 35. With t=box_spacing/100, compute in world units:

- padding p=3+7t, width w=134+32t;
- header baseline h=p+8, title baseline a=h+15+3t;
- shape baseline s=a+13+3t, detail baseline d=s+11+4t;
- footer inset 3, base height d+p+3, socket top a+5, row gap 14+5t;
- minimum height=max(base height, socket top+10+port-row-count×row gap).

Each node may save optional width,height in world units. Effective width/height
is max(stored preference, computed minimum); absent values mean automatic sizing.
Both preferences must be finite positive numbers ≤100000, excluding booleans.
They persist in the node JSON, not the user config or menu geometry. Adding ports
can increase effective height without rewriting a smaller stored preference.
Changing compactness recomputes minimums, not saved positions or preferred sizes.

Port-row-count is Concat.sources.length+1 (including its free input socket),
router port count, or 1. A single socket is at height/2; multi-socket y is
socket top+(index+0.5)×(height−socket top−8)/count. Output x=effective width; input x=0.
The detail baseline uses actual height−padding−footer inset, so growing multi-port
boxes keep the footer at the bottom. At level 100 a one-port box is 166 × 80;
at 35 it is 145.2 × 64.4; at 1 it is 134.32 × 56.24. These tighter baseline gaps
avoid large empty middle areas. Level 1 is densest; fonts remain header 9, title 14, shape 10,
detail 9 world units. Top-left model positions and graph semantics never change.
Renderer, sockets, wire endpoints and Fit must share these metrics, not duplicate
fixed widths. Long titles/shapes/details truncate with full title hover text.
Reserve 14 units at the footer's right for the resize grip. For pertinent learned
operations, display bias state before truncatable statistics: Linear starts
Bias on/off; Conv2d does too. Its parameter count is
C_out×(C_in/groups)×kH×kW+(bias ? C_out : 0). Linear's parameter count is
d_in×d_out+(bias ? d_out : 0), using last-axis
widths, not flattened sample counts. Earlier axes reuse the same weights. Inspector
shows A [d_out,d_in], and either b [d_out] or Bias off · no b. Do not imply that
non-bias operations contain bias or collapse a module's heterogeneous internals
into one misleading bias flag.
Distinct 20-unit vector type icons occupy top-right, scaled from 24-unit paths.
They are accessible, focusable folding controls. Clicking an expanded box's
icon collapses the box to a compact icon-and-summary body; clicking a compact icon selects it
without expanding. Dragging that compact icon moves the block with the same
world-coordinate and Ctrl-snap behavior as an expanded block. Double-click
restores its details; Enter/Space toggles folding on a focused icon. Dragging
must not expand or collapse it, and sockets remain separately clickable.
Plain Space also toggles the selected block while the diagram has focus; focus
need not be on that block's icon. One initial press gives one history edit;
autorepeat is consumed without repeated toggles. A wire-only/empty selection,
running job, active canvas drag/wiring, dialog, editable field or focused control
does not trigger this global block shortcut. Modified Space is left alone.
The existing focused-icon Enter/Space behavior is retained.
Compact width is 72 world units; height is at least 54 for Linear/Conv2d/LSTM or 44 otherwise,
increasing to keep multiple ports distinct. Small 9-unit monospace text shows
input/output shape, Linear last-axis in→out widths, activation name or type-specific
summary. Linear/Conv2d/LSTM also have an 8-unit Bias on/off line. Summaries truncate after
13 characters with full hover text; missing inferred dimensions show ?. The type
icon sits at the compact body's top-center and remains directly draggable.
Folding preserves preferred expanded sizes, tensor
semantics, datasets and results. Its boolean collapsed flag persists with the
block and participates in undo/redo. Training disables folding and movement.
Bézier bend is at least 50
units, otherwise approximately 45% of horizontal separation. View wheel zoom uses
the pointer as anchor; manual graph zoom is clamped to 0.05–2.5; Fit caps initial
scale at 1.3 but may go below 0.05 to keep the graph inside the free rectangle.
These numbers are defaults, not domain laws. Menu CSS and diagram SVG text must
remain independently tuneable when adopting larger readable defaults.

Every box has a 14×14 bottom-right SVG resize target with two diagonal marks,
nwse-resize cursor, accessible label/disabled state and keyboard focus. Drag in
inverse canvas/world coordinates; do not change top-left. Active snapping aligns bottom-right
to the 24-unit world grid, raising a snapped corner as necessary to satisfy the
minimum. Arrow keys change a focused grip by 8 units; Shift or Ctrl uses 24, and
Ctrl inverts the block snap default. The Inspector has no removal or diagram width/height controls.
The block context menu's Automatic size deletes both size preferences, preserving
position, folding and computation. It is undoable and disabled during training;
the dispatcher rechecks the captured model/block IDs and busy state. Repeating
it on an automatically sized block makes no edit. Do not start inspection on
grip double-click.
Capture the persistent canvas, not the re-created SVG node, during drag. Redraw
nodes and wire endpoints together. Release/cancel/lost-capture/blur ends the
gesture, clears resize CSS state and persists one edit if moved. Training disables
resizing. Saved sizes participate in workspace undo/redo, but preserve code,
contracts, datasets and learned results because they are presentation-only.

Single click selects but need not open a hidden Inspector; double-click opens it.

### Building-block palette help

Right-click a Building blocks palette button, including its nested icon or label,
to get About <button name> as the first native context-menu action, followed by
a separator and the general actions. This is not a diagram-block context menu;
it neither selects nor deletes a diagram block. `contextAt` returns
`{palette:{id,name}}` instead of a block descriptor for this target. Tensor
buttons have stable DOM IDs `add-tensor-<type>`.

About dispatches `about-building-block` with the palette button ID. Recheck that
the target still exists in the library. Show a themed, scrollable HTML dialog
with a focusable Close button and native Escape cancellation. App shortcuts do
not intercept an open dialog. Help remains available during training.
`building-block-help.js` supplies all 15 descriptions, using the same concrete
tensor-operator explanations/defaults as the Inspector. Activation and reference
help reflect the currently selected palette options. Explain shape/axes, learned
parameters, insertion behavior and important differences or restrictions. Define B
as batch size. Linear's equation uses the existing local MathJax renderer with a
plain-text fallback. Help paragraphs inherit the user base font and theme;
render names/text through textContent, never raw external HTML. Reading or closing
help must leave graph, dataset, results, SQLite bytes and undo history unchanged.

### Bulk block layout tools

Layout is a floating major menu with a launcher button and native context-menu
entry. Its independent details sections are All Blocks Collapsed (initially open)
and All Blocks Expanded (initially closed); both can be folded/expanded using
standard accessible HTML summary controls. Each contains a layout style selector,
direction, between-layer/between-block gaps, Rearrange blocks checkbox, Fit after
action checkbox and its Minimize all/Expand all button. Layered is the only
implemented style; future styles are additions to this explicit dispatch boundary.

Collapsed defaults: LR, layer gap 48, block gap 16, rearrange=true, fit=true.
Expanded defaults: LR, layer gap 80, block gap 32, rearrange=true, fit=true.
Gaps are integers 8–1000 world units, independent of diagram/UI zoom and padding.
Directions LR and TB mean left-to-right/top-to-bottom node ranking; ports retain
their existing left/right positions in both. Style/profile controls do not alter
port identity or compute order. Current control values apply on each button press.
Expanded profiles optionally accept strict boolean `parameter_sizing` (default
off; missing legacy key means off). Enabled Expand all collapses zero-parameter
blocks and expands positive-count blocks. The least positive count uses readable
minimum height; linear relative scaling reaches 4× minimum at the largest count,
or all minimum when counts tie. Module instances include internals. Widths are
unchanged. This intentionally replaces positive blocks' preferred heights;
undo/Automatic size can restore/reset them. Native exact decimal counts are read
only for a validated matching graph/workspace instance; stale replies never
replace a newer graph. Ordinary layout still supports drafts. See
[layout.md](layout.md) for the formula, count grammar, CLI and precise limits.
Save layout preferences updates both profiles in the current workspace's
`metadata.layout_preferences` JSON, as one session undo entry. Explicit Save
workspace writes the `.vtw` snapshot. Full state carries `workspace_id` and
`layout_preferences` outside per-user `display`. Tabs refresh controls when the
workspace identity/profiles change, including undo/redo, but unrelated state
updates do not discard unsaved control edits. Missing metadata uses independent
defaults. Legacy global `layout` is inert, never imported. Updates require a
workspace and no active training; no-op changes create no history entry.

The offline bounded Layered kernel uses SCC condensation for cyclic drafts,
longest-path ranks, four alternating port-aware barycentric ordering sweeps and
size-aware, centered non-overlapping rank placement. It does not import ELK/Dagre
or guarantee crossing-free/optimal routing. Concat source arrays, tensor packing,
graph arrays/edges and reusable module contents are not reordered or unfolded.
Disconnected/cyclic drafts remain drafts; layout never repairs computation.

Prepare the changed graph on a private clone, applying all compact flags before
measuring layout dimensions. Reject bad preferences/endpoints or coordinates
outside ±100000 without replacing the original diagram. Apply state, geometry and
optional fitting as exactly one workspace/history edit. Undo restores previous
folding and positions; the session camera is preserved, not restored from history.
Ordinary expansion retains preferred dimensions; parameter sizing replaces
positive-count preferred heights deliberately, with undo restoring them.
Keep current scientific results/datasets/bindings intact. Do not allow either
button during training or an active block gesture. Fit honors floating overlays
and may request more visible room when panels cover the canvas.
Single selection may also target a directed wire, keyed by (from,to,from_port),
not by a mutable array index. Group selection is deliberately not implemented.
Unmodified D deletes the selected box or precise wire through normal history;
Delete/Backspace act with canvas focus. Do not intercept typing/contenteditable,
modals, Ctrl/Alt/Meta-modified or repeated keys or active drags; training disables mutations.
Deleting a box retains the existing minimum-two-block and bypass rules. Deleting
a wire removes only its exact identity and corresponding ordered Concat key.

SVG destination handles occupy a separate world overlay after boxes, at x−11
world units from each input socket's endpoint, radius 4 (selected radius 6).
They select the wire and start a destination-only drag, capturing the persistent
canvas. The live preview is separate from saved edges. Pointerup hit-tests the
actual screen coordinate with elementFromPoint because captured event.target is
the canvas. Only an empty input or the final free Concat socket accepts a drop.
Preserve from/from_port and unrelated fanout. Validate against a graph-only copy;
reject absent endpoints, occupied inputs, self-links and cycles atomically.
Remove the old Concat key without sorting other inputs; append at the new Concat.
Dropping blank or on an invalid socket, Escape, pointercancel, lost capture, blur,
or changing architecture cancels unchanged. A valid release persists exactly one
computational/history edit and invalidates current results. Shape compatibility
errors remain explicit saveable drafts; no automatic reshape is inferred.
Update selection classes in place so ordinary wire double-click deletion remains
deliverable. The original SVG endpoint and other model/panel gestures are separate.

Empty-canvas clicks and panning preserve box/wire selection.
Escape prioritizes cancellation of active canvas drag or armed wiring. Otherwise
it clears selection; with no selection it hides all floating panels and shrinks
the menu pack, preserving positions. Typing controls, open dialogs, modified or
repeated keys keep their own Escape behavior. Selection/hiding creates no edit.

Right-click selects before the app context menu opens, without beginning
drag/wiring; cancel any armed wire. The native WebEngine menu callback first
calls loom.contextAt(localX/pageZoom,localY/pageZoom), then opens QMenu only after
the JavaScript callback. Discard stale callback request IDs. This compensates
for native context events whose preceding page pointer event may not arrive.
contextAt returns null outside a box, otherwise {id,modelId,canDelete,canResize} after
selecting it. On a box the native menu begins Inspect block, then Delete block,
then Automatic size,
then a one-pixel horizontal separator. General graph/panel actions follow;
a second separator precedes workspace actions. General groups remain alphabetical;
the block section is intentionally ordered by inspection before destruction.
All block actions capture the exact IDs. Inspect dispatches inspect-block,
rechecks current model/block existence, clears port/wire selection, refreshes
selection classes and Inspector, and opens inspector-panel without persistence
or history. Inspection is available during training; stale targets are ignored.
Delete is disabled during training and at the grammar's minimum two
blocks; no deletion action appears on blank canvas or home. The dispatcher also
rechecks the current architecture ID, block existence, busy state and minimum
size, so later selection/navigation cannot redirect deletion. Removal uses the
existing workspace history path, not direct SQL. Boundary deletion can leave a
saveable invalid draft. Inspector no longer offers Delete architecture; the
existing CLI/backend collection-deletion capability is unchanged.
Keep the original hit element alive between clicks: immediate full SVG replacement
can prevent double-click delivery. Current drag code updates selection classes
in-place on pointerdown. Rebuild after actual movement/state changes instead.

Visual PyTorch specifics, **not generic reuse requirements**: tensor shapes, Linear/
Activation/Reshape, Concat, Splitter, Replicator, Module/Recurrent reference interfaces, training
contracts, MSE, constrained runs. Reference normal double-click inspects; Ctrl/
Meta-double-click opens the referred graph in a new tab. Inspector also has Open
details/Open in new tab actions. Socket click wiring and arbitrary DAG checks live
in the domain adapter, not in the floating-menu manager.

Recurrent references use the same floating Inspector and loop-type icon, not a
new UI scheme. Their physical input sockets represent distinct explicit sources
in sequence/state/lengths/mask/reset binding order. A deliberately shared source
has one socket with all its argument roles in the tooltip. Returned sockets are
sequence outputs then final states. Socket tooltips show role/identity/individual
shape. Icons fold, compact icons drag, double-click expands, and Ctrl/Meta body
double-click opens the transition in a new tab. Selected-port insertion defaults
must use that tensor shape, not the packed aggregate. The guarded branch-creation
modal makes all input roles/state outputs explicit and preserves existing work;
its native ownership/closure tokens and precise domain semantics are specified
in [recurrent-graph-contract.md](recurrent-graph-contract.md#qt-authoring-and-agent-parity).
Keep this pure domain adapter (`recurrent-editing.js`), backend builder and Qt
bridge separate from reusable floating-panel mechanics.

Inspector also opens an advanced in-place interface modal. It prefills exact
JSON bindings, shows stable signatures/source IDs and retains rejected drafts.
Changing a reference never guesses new bindings; connected return ports cannot
be removed implicitly. Read/write tokens fence both current and candidate
closures plus the active workspace instance, and obsolete modal responses cannot
replace newer sessions. One accepted edit is one undo step; exact no-ops change
nothing. Preserve unrelated drafts rather than "repairing" them behind the
user's back. These are reusable interaction principles, not a new panel scheme.
Domain schema and affected-path proof: [interface editing](recurrent-interface-editing.md).

The concrete LSTM domain adapter reuses this scheme with a distinct cell/gate
icon and amber semantic token, not hard-coded dark fills. Its input sockets are
the distinct source keys of sequence/hidden/cell, then optional lengths; its
three output sockets are sequence/hidden/cell. Inspector must distinguish cell
and projected hidden widths and show the explicit state batch-axis adapter.
Both expanded and folded blocks always disclose bias. Compact icons remain
draggable, all sockets remain distinct, and selected-port insertions use that
port's shape. Its standalone palette action opens a typed creation modal;
Inspector opens an in-place constructor/source/owner editor. Preview distinguishes
hidden `[B,L*D,Q]` from cell `[B,L*D,H]`; initial states/lengths are explicit.
Preserve existing IDs/outgoing wires rather than guess compatible replacements.
Unproven source shapes are labelled, native errors keep editable field values,
and workspace-instance/graph/reference/generation guards fence stale responses.
Only one prepare/apply is pending; a true no-op leaves bytes/history unchanged.
Cancel fences UI delivery, not rollback of a previously authorized native write.
Keep existing panels/typing/history behavior rather than add a separate recurrence
UI. Exact closed routes/options and remaining concrete-only limits:
[LSTM authoring](lstm-authoring.md) and [forward contract](lstm-graph-contract.md).

## 8. Visual system and accessibility

Default palette: dark green/blue-gray stage; mint primary actions/valid state;
blue wires, purple learned operations, amber references/routing. Neutral text and
visible symbols/labels communicate states without relying only on color.
Rounded menu corners are 11 units; pack corners 12; borders/shadows distinguish
floating rectangles without allocating gutters around the canvas.

One compact top header combines brand/navigation/workspace identity and the
architecture toolbar. Responsive rules hide subordinate brand/workspace text at
narrow widths; crucial actions remain. There is no second toolbar canvas row.
Native Qt tab strip, menu bar and dropdown/context menus follow the active theme;
the strip hides when only one tab exists. A fixed stylesheet overrides QPalette,
so explicitly replace native menu styles when changing theme, including startup.

Readable defaults: base UI font 14px, page zoom 1.15; many secondary labels use
base minus 1–3 pixels, not the old 9px-heavy controls. Options permits base font
12–28px and UI zoom 0.5–3; Ctrl+/Ctrl− updates every tab and persists. Menu spacing
is an integer 1–10, default 4. Set CSS custom property --menu-unit to that many
CSS pixels. Palette padding is u vertically and 1.25u horizontally, margin 1.25u,
gap u, minimum height 24+2u. Other content uses 0.5u–1.75u: section padding 1.5u
by 1.25u; input padding 0.75u by u; launcher button interior padding 0.75u and
icon-to-text gap u, minimum height 1.4em+1.4u. Launcher buttons have inter-button
gap u; their list has side/bottom padding 2u and top padding u. Inter-button
gaps and side/bottom frame margins are doubled relative to the previous layout
(0.5u and u respectively); button interior padding is not doubled.
Do not scale fonts, frame recovery areas or heading drag handles.
Expanded launcher width/height remains content-driven for large fonts. Old
Compact/Spaced config values load as 4/10 without a write; explicit Save persists
the numeric value. This is a user-level appearance setting, not graph data. NN box
spacing is the independent numeric setting above. These settings are distinct
from diagram pan/zoom. Display preferences has Preview/Save/Cancel: Preview
applies all appearance values without writing config or closing the dialog;
Cancel restores the original values; Save persists selected values. Failure to
save restores previous appearance. Minimum native dialog width max(480,30f) and
size grip keep large-font controls readable. OS-decorated titles may still need
normal-desktop verification. Cross-project adopters should use
their own config path/name; ~/.tensorloon/config.json is Visual PyTorch's retained
compatibility path. QSettings VisualTorch/TensorLoom preserves recent files.

Use actual buttons, focus-visible outlines, labels/aria-controls/expanded/pressed,
keyboard pack restoration and resize/scroll controls. Hide uses hidden attribute,
not merely opacity. Respect prefers-reduced-motion for scrollbar fades. HTML
confirmations use task headings/aria-labelledby, Cancel focus and reset returnValue
on every invocation; do not expose file-origin paths as dialog titles.

Browser Save page/View page source/Reload actions are removed by a native Qt
custom context menu. App actions call a narrow command dispatcher. This is not
a custom left-click pop-up menu. Validation banners anchor upper-right, 16 units
from stage right and 12 from top. Notifications sit bottom-center above menu
layers: 5500ms automatic lifetime, opacity transitions 250ms in/out, then hidden;
an accessible × button dismisses early. New notifications cancel old close timers.
Notification text is pointer-transparent, so a low-positioned panel button remains
usable underneath it; only the visible close button accepts pointer events.
Respect reduced motion. Use a separate message span so updates retain the button.

Canvas pan/zoom/Fit are session-only and never write the workspace or add history
steps. Every architecture open ignores legacy camera metadata and centers/fits
its current geometry. Ordinary acknowledgements preserve the tab camera;
workspace-instance changes refit. GUI model edits submit neutral legacy `view`,
not the active camera. Old files are not rewritten merely by opening them.
Global interface zoom is still persisted independently as a user preference.

Themes share one geometry/interaction system with semantic colour tokens in
app/themes.py: Aurora, Bauhaus, Blueprint, Cyberpunk, Dark, GitHub, Light, Math,
Matrix, Metallic, Monokai, Neon, Obsidian, Oceanic, Retrofuturism, System, Terminal.
Always sort theme choices alphabetically. Apply the same rule to unordered
browsing lists: architecture/reference names, activation choices, block choices
and launcher menus. Keep logical groups and coupled controls together; preserve
selected values and explicit defaults. Never alphabetize meaningful sequences:
recents, layer execution, tensor/boundary packing, Concat sources, split sections,
port arrays or ordered edit batches. Conventional File/Edit menu action groups
also retain their established task ordering.
There are 17 choices: 16 palettes and System. Metallic uses steel/gunmetal semantic
tokens plus theme-scoped linear gradients and inset highlights on web chrome,
pack/panels and closed launcher buttons; selected/hover states remain explicit.
Math uses a serif stack for body prose, labels and SVG titles: STIX Two Text,
STIX, Noto Serif, DejaVu Serif, Liberation Serif, Georgia, generic serif. Native
Qt chooses installed families in that order with a Serif style hint. Explicit
code/tensor/table monospace and MathJax's own formula fonts are not replaced.
Capture the baseline Qt font before first theme application; restore a fresh
copy on every non-Math theme so serif does not leak into later themes or Cancel.
Use the configured pixel size in either case. Do not download/install fonts.
See [QFont family fallback and style hints](https://doc.qt.io/qt-6/qfont.html)
and [installed font families](https://doc.qt.io/qt-6/qfontdatabase.html).
These are locally designed inspirations, not official theme distributions.
System resolves Qt's reported Light/Dark scheme and listens for changes; unknown
falls back to Dark without changing the saved System choice. Light/GitHub/Math/
Retrofuturism/Bauhaus use browser color-scheme:light; others use dark. Keep one
token source for native and web backgrounds, controls, text, accents and node
fills. Platform title bars/file choosers may retain OS styling. Palette tests
are not a full contrast/accessibility audit.

I/A/B/E/H toggle Inspector/Architectures/Building blocks/Experiment/Help, ignoring
editable fields, modified/repeated keys and modal dialogs. Panel shortcuts also
leave focused buttons, links, disclosure controls and ARIA controls alone.
Help is a regular floating, draggable, resizable panel with a short drawing guide
and a semantic shortcuts table. It starts closed; the bottom-of-canvas shortcut
caption is removed (the status footer and validation chip remain). Help content
scrolls through the same overlay scrollbar as the other panels, uses theme tokens
and configured typography, and wraps within narrow resized frames. Help geometry
is tab-local: version-1 workspace panel presets retain their exact five-panel
grammar; Help is excluded from capture and closes on preset restoration/reset.
It still participates in menu-layer drag, stacking, Fit occlusion and hide-all.
Section links scroll inside Help without changing the architecture URL hash.
Learning-resource links use an explicit desktop allowlist and open in the system
browser only on user activation; they cannot navigate the trusted editor or
provide arbitrary URLs. Initially linked resources are official PyTorch
[Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
and [torch.nn](https://docs.pytorch.org/docs/stable/nn.html), checked 2026-10-11.
External documentation is not an assertion that every described operator is
implemented. Local guides/shortcuts work offline; linked guides need internet.
Keep sections extensible for future app documentation and worked NN guides.
Ctrl+Z undoes workspace
edits; Ctrl+Shift+Z/Ctrl+Y redo; text fields retain native text undo. Native Edit
actions dispatch the same commands without stealing text-field shortcuts.
Domain history is session-only SQLite snapshots (50 steps/128 MiB combined),
not floating-menu/display preferences. No history edits during training;
preserve run evidence for surviving restored architectures. A different product
must choose its own transactional history implementation, not blindly reuse .vtw.

The plain browser-page Find button, context-menu entry and Ctrl+F binding are
removed; no replacement architecture search is claimed.

Group selection is explicitly NOT implemented. The owner is considering ordinary
selection versus Ctrl-selection/drag; a future decision may replace current Ctrl
snapping. Do not infer or silently implement a modifier reassignment from this handoff.

### Mathematical content

MathJax is local/offline TeX+AMS → SVG. app/web/math.js isolates the available
legacy 2.7.9 API behind LoomMath.ready, typeset(element), set(element,tex,{display}).
Use typeset after adding math containers, or set for replacement; it removes old
jax before serializing new work. Only class loom-math is processed; normal labels,
code and SVG diagram text are excluded. Supported delimiters: inline `\(…\)`,
display `\[…\]` or `$$…$$`. Output inherits theme colour. Inspector formulas use
trusted application TeX, not unescaped workspace labels. Safe filters URLs,
classes, CSS IDs, styles and TeX require; buffer/macro limits also apply.
Math renderer menus are disabled so they do not reintroduce source/browser menus.
This is not a hostile-TeX security sandbox or full accessibility certification.

Keep the full selected vendor runtime closure and licenses; provenance/pins are
in app/web/vendor/mathjax/PROVENANCE.md. Legacy was available locally when network
access prevented newer acquisition. A future modern MathJax upgrade requires
an adapter/config review, not simply replacing the file. Relevant primary docs:
[v2 configuration](https://docs.mathjax.org/en/v2.7/config-files.html),
[v2 dynamic loading](https://docs.mathjax.org/en/v2.7/advanced/dynamic.html), and
[modern local hosting](https://docs.mathjax.org/en/v3.2/web/hosting.html).

## 9. Reuse boundary and source map

| Source | Responsibility | Reuse guidance |
| --- | --- | --- |
| app/web/panels.js | Geometry, pack/menu state, gestures, overlay scrollbars, Fit free rectangle | Domain-independent core; adapt DOM bindings carefully |
| app/web/style.css | Dark palette, floating rectangles, pack, scroll/thumb/controls, readable sizing | Copy relevant rules, not obsolete original fixed-grid CSS |
| app/web/themes.css; app/themes.py | Shared semantic web/native colours and theme typography | Copy together, parameterize theme choices and font stacks |
| app/web/type-icons.js | Distinct trusted vector fragments for box types | Replace domain types; preserve top-right no-hit geometry |
| app/web/notifications.js | Timed fade/dismiss lifecycle | Domain-independent adapter; keep stable message/button DOM |
| app/web/math.js; app/web/vendor/mathjax/ | Offline math adapter and licensed runtime | Copy the complete closure; audit legacy version/upgrade |
| app/web/index.html | Concrete DOM with linked launcher/menu IDs | Replace content/labels while preserving required bindings |
| app/web/editor.js | Canvas rendering, domain inspectors, persistence/Qt bridge | Extract generic drawing mechanics; do not import Torch semantics blindly |
| app/web/graph-tools.js | Diagram port geometry and Visual PyTorch graph mutation | Some geometry reusable; mutation rules are domain-specific |
| app/qt_style.py | Native tab styling/auto-hide | Optional for Qt adopters |
| app/context_menu.py | Qt app menu replacing browser actions | Adapt command list to your domain |
| app/config.py | Atomic user appearance/RAM-preference JSON | Optional; rename app-specific config path |
| app/history.py | Bounded session SQLite snapshots | Domain-specific workspace history, not panel-manager state |
| assets/tensor-loom.svg; app/identity.py | Visual PyTorch V-above-T circuit identity, visible name and stable legacy launcher/icon IDs | Do not assume another product should use this brand |
| tests/panels.test.js | Geometry/state/gesture tests with small fake DOM | Port with menu manager; not rendered-layout tests |

Minimum panel-manager DOM contract:

```html
<main id="editor">
  <nav id="menu-pack">
    <div class="pack-heading"><span>TOOLS</span><button id="pack-toggle" aria-label="Collapse tools"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5V3h13v13h-2"/><rect x="3" y="8" width="13" height="13" rx="2"/></svg></button></div>
    <div id="pack-menus">
      <button data-open-panel="example-panel" aria-controls="example-panel" aria-expanded="false">Example</button>
    </div>
  </nav>
  <aside id="example-panel" class="floating-panel" hidden>
    <div data-drag-handle>
      <span>EXAMPLE</span>
      <button data-expand-panel>⤢</button><button data-fold-panel>×</button>
    </div>
    <div id="example-scroll" class="scrollport">Project-specific controls</div>
  </aside>
</main>
```

Load panels.js after that DOM, initialize(document), and retain its reset/open/
toggle/available API. The manager creates resize grips/overlay rails; do not duplicate
them. Bind the reset command separately. The current label mapping in initialize
knows Visual PyTorch IDs; an adopter should parameterize labels, not preserve that
conditional accidentally. Generic panel titles/controls require corresponding CSS.

Copy the complete runnable closure and record source/provenance under the receiving
project's workflow. A link to this campaign is background context, not an executable
dependency. No reusable packaged library or automatic cross-project installer is
claimed here. Preserve this handoff with the copied source, then record local changes.

## 10. Acceptance checks for another project's implementation

Test pack collapse/restore with pointer AND keyboard; opening any combination of
menus; expansion/restore after close; arbitrary empty-area drag without breaking
text/controls; off-edge recovery; free/Ctrl drag and free/Ctrl resize anchors;
Shift/Ctrl+Shift layer translation including closed menus; observer scrollbar
updates after content changes; focus returns; Fit when menus cover the canvas;
double-click delivery after selection; right-click select-before-popup at non-unit
page zoom; port-specific wire deletion; independent menu/box compactness;
box metrics/ports/Fit at levels 1/35/100, saved per-box overrides and free/Ctrl
resize with history/automatic reset; visible bias state and last-axis parameter
counts; alphabetical browsing without semantic reordering; all themes at startup/live switching;
Preview/Cancel/Save without premature writes; page zoom persistence; 250ms toast
fade/dismiss/replacement; offline math startup and dynamic replacement; native
one/two-tab states if using Qt. Match keyboard/text undo ownership and history bounds.

The source tests pass for geometry/state and compiler-independent contracts. A
fake DOM does not validate fonts, pixel geometry or real Chromium event routing.
Earlier WebEngine startup was blocked; after access changed, tests/qt_ui_smoke.py
passed against the real HTML/editor/Qt bridge and a private in-memory workspace,
retaining Chromium sandboxing. It checks top-left/reset placement, intrinsic
pack overflow, numeric gaps and serif/monospace at three font/spacing combinations
in GitHub/Math/Metallic, real Qt-mouse resizing through Chromium pointer capture,
saved geometry/ports/history/reset, Ctrl keyboard corner snapping, bias state,
alphabetical lists, explicit-target deletion and busy guards. Undo/redo includes
a deleted Output. Math/Metallic captures were visually inspected. This does not
prove every mouse path, all palettes or native desktop integration. MathJax's
adapter tests still use a stub Hub, not an asserted rendered-equation audit.
Normal-desktop mouse, dialog, readability and Alt-Tab checks remain necessary.
Never disable browser security just to declare a visual test passed.

## Domain evidence and dataset admission (Visual PyTorch extension)

Graph validation and run admission are distinct. The Experiment menu's Train
button is disabled for incompatible/unbound/invalid datasets even when graph
geometry and compilation are valid. Its Column bindings disclosure shows both
recorded and current declared input/output IDs, sample shapes and packed x/y
offsets. It opens automatically for a binding problem. A Use current port order
button appears only when valid existing numeric rows can be explicitly rebound.
The native confirmation has a human app/action title and Cancel as default; a
cancelled or stale confirmation does not edit. Accepted rebind is one undoable
operation, does not move numeric values and retains historical runs. Training
disables the action. This is domain-specific content inside the existing floating
Experiment menu, not a new menu architecture.

Result labels distinguish current compatible checkpoints, unverified legacy
results, stale/invalid evidence and no result. Legacy metrics have an explicit
Unverified label; stale/invalid evidence cannot supply displayed current loss
curves. Results explain update counts, stop reason, post-update/eval timing and
optional analysis availability separately from successful execution. The PyTorch
forward view/export is explicitly a fresh-weight architecture template, not
trained inference; the CLI/API supplies validated trained export. Do not copy a
domain result label into a generic reusable interface without its actual admission
contract. Native/WebEngine tests assert these distinctions through the real bridge.

Design reference: [Qt tool-palette controls](https://doc.qt.io/qt-6/qdockwidget.html)
informed movable/closable/expandable affordances. The implemented menus are HTML,
not QDockWidget instances. The pack, independent layer gestures and snapping
anchors are the owner's requested interaction design, not borrowed native behavior.

## Typed scientific drafts (Stage C Visual PyTorch extension)

The floating Experiment/history tab hosts typed dataset, task and experiment drafts. Reuse
their interaction pattern, not the domain grammar by accident: stable identities,
exact historical base distinct from expected current head, private unsaved text,
read-only validation, explicit immutable admission and separate file Save. Context
suggestions must never replace exact selections or infer bindings by list order.
Reordering a semantic chain is an explicit up/down operation, not alphabetical
presentation sorting. Numeric parse errors stay visible and block destructive
rerender/admission; refreshing suggestions must retain that text. Typing clears
old compatibility results before blur. Checks are visibly scoped: intrinsic object
validity is not interface compatibility, and neither proves execution/resource
admission. Asynchronous replies from another workspace/draft are ignored.

The accepted panel layer/keyboard/typing and history behavior remain unchanged.
Task/dataset forms use theme colors, base-font controls and responsive grids inside
the same scroll container. Task transform action buttons use intrinsic compact
width, not equal-width expansion. All task bindings are currently rendered, while
dataset cells page six fields/twenty records: neither is a full transport bound.
Experiment selections use explicit exact references, ordered axes, independent
seeds, typed split/optimizer/budget controls and reasoned comparison exclusions.
Refresh must preserve selected identities, including unavailable historical refs.
Comparison-mode changes must not discard conflicting whitelist entries. External
observation IDs are literal lines, not comma-separated or silently deduplicated.
Train-free draft preview pages 25 coordinates with no Run buttons, clears on edits
and returns exact decimal Cartesian counts. Retain first, then run explicitly.
All experiment axis cards currently render; paging is not a metadata/transport or
RAM bound. Full C convergence remains unfinished.

For read-only retained objects, reuse a separate bounded inspection surface,
not the mutable authoring draft or a collapsed copy of the entire JSON. Metadata
choices page alphabetically and preserve an exact historical pin outside the
current page/search. Typed summaries must declare their projection scope, never
masquerade as a complete editable specification. Tree navigation uses immutable
member ordinals, preserves scientific array order, marks partial keys/values and
provides explicit full-key/scalar paging. Show actual cursor/range and retain
visited offsets; byte-limited pages need not have the requested row count.
Use literal text nodes and local table/scalar overflow with theme/base fonts.
Disable/guard duplicate pending navigation. Fence late success AND error delivery
by workspace and selection/generation; a notified same-UUID reload must also
invalidate pending contents. Verify returned revision/path/part/offset before
display. Scope limits to response/DOM bytes/counts; full registry validation and
draft transport are separate costs. The Visual PyTorch realization and reusable
contract are in [scientific-browser.md](scientific-browser.md).

For stored-reference authoring controls, reuse independent bounded revision
pickers, not one complete metadata index copied into every selector. Preserve
chosen exact pins across search/page/refresh; represent missing pins separately
from valid metadata. Explicit experiment selections must not be guessed from
list position. Search tools are siblings of a single-control select label, with
their own accessible names, compact grid/range and full exact-ID/hash tooltip.
Fence pending successes/errors by owner/workspace/generation/disposal, validate
returned kind/query/cursor/pin, and guard repeated navigation. Suggestion refresh
must not rebuild/discard incomplete unrelated fields. A notified same-UUID reload
also invalidates suggestions without replacing draft refs/text. This does not
page local draft axes or bound whole draft transport. Task field/port suggestions
now use independent bounded datalist pages, literal search, exact-ID inspection
and visible missing pins. Keep original scientific ordinal alongside alphabetic
browsing, and label partial shape/axes previews as previews with actual rank.
Do not send observations/vocabularies/arbitrary metadata merely for suggestions.
Search/paging must not rebuild/flush binding controls or discard invalid text.
The shared GUI/CLI contract and unbounded native-cost limits are in
[task-context-pages.md](task-context-pages.md). The executable
realization and reusable limitations are in
[authoring-revision-choices.md](authoring-revision-choices.md).
For large mutable documents, distinguish an owned editor session/sequence from
an immutable scientific revision. Read only projected pages; send atomic ordered
patch batches against that sequence rather than repeatedly submitting complete
drafts. Bind actual owner/working-copy/connection identity on the host, retaining
unsaved inspection after switches but rejecting admission into a replacement.
Close explicitly or on owner teardown; failed checks must preserve bytes.
Task binding/transform cards must retain scientific ordinals, independently page
pipelines and return complete literal JSON or explicit staged-value markers.
Send small typed values as native-parsed literal text rather than JavaScript
Number conversions; larger values use the same parent session's child editor.
Resolve named properties against the current native document after each ordered
operation; removing optional keys must not shift a later edit into another key.
Flush prior queues and reload ordinal paths before structural changes. The actual
task form implements this pattern; its exact protocol and tests are specified
in [typed-task-sessions.md](typed-task-sessions.md). Bound visible binding cards
and each visible pipeline independently. Update Back/Next from current navigation
history, not cached flags from before history was advanced. Settle parallel reads
before lifting the pending-command guard even if one fails early. Retry must
reuse the owner/session and retain invalid text; reject wrong/stale page and
admission identities before painting. Small properties use compact typed fields;
explicit incomplete large markers offer a same-session staged-editor button.
Returning to a UUID alone cannot resume editing: prove actual native owner,
Workspace, connection and sequence. Notified reload stays stale until explicit
cancel/reopen. This is not durable recovery or a universal ABA detector.
Preserve scientific order and invalid local text. A wire limit is not a bound on
whole-host parsing/copies/history/validation; long values need separate editing
and resource safeguards rather than truncation. The current tested backend/Qt
service is in [owned-authoring-sessions.md](owned-authoring-sessions.md). The actual
Qt value editor stages raw JSON in one owned temporary buffer, separates buffer
revision from document sequence, uses UTF-8 byte cursors and explicit whole-value
acceptance, and preserves malformed text/original values. RAM admission estimates
precede full parsing/undo capture, not universal allocator bounds. Dirty navigation,
exact identities, pending commands and reload-fenced replies need independent
tests. Reuse the precise protocol in [chunked-draft-values.md](chunked-draft-values.md).
The actual dataset form now uses projected literal descriptors, stable-target
queued edits and a linked child editor in the same session. Capture focused
dirty text before disabling controls; reject late identity/sequence/cursor replies.
Flush queued small edits before resolving a current ordinal by stable IDs; object
key insertion order is not identity. Child Finish/Discard return a verified
advanced/unchanged header, never retain/discard the parent. Keep validation scoped
to the owning form, and clean discarded controls on explicit Cancel. Offer page
retry without losing text or creating another session; initial read failure must
not enable actions needing a page. Batched edits are atomic per batch, not an
entire multi-batch form transaction. Large markers are not truncated values.
Reserve readable table ID columns and local horizontal scrolling instead of
squeezing headings letter-by-letter; test rendered geometry across font settings.
The exact reusable protocol is in [typed-dataset-sessions.md](typed-dataset-sessions.md).
Actual experiment forms now also use owned literal cards/edits and staged children.
Keep notification signals from shadowing
native QObject event methods while preserving the advertised channel name.
See [the executable experiment GUI contract](experiment-gui.md) for action schemas,
precise scoped compatibility, failure behavior and actual Qt test evidence.

Bound each mutable local axis independently, not just remote registry choices.
Retain original ordinals, exact out-of-page pins, incomplete entries and duplicate
selection diagnostics while sorting browsing metadata. Named nested fields must
resolve after earlier key removals; array ordering is computation, not decoration.
Projected prospective candidates should expose identities/status/explicit partial
diagnostics/counts without repeatedly transferring full split manifests, role IDs
or training protocols. No-fit preview still has native validation/manifest costs.
The tested native primitives and closed contracts are in
[typed-experiment-sessions.md](typed-experiment-sessions.md); the actual experiment
controller now adopts them. Reference dropdown identity must ignore JSON key order;
pairing values remain literal IDs, not JSON strings. Disable controls as soon as
the owned header is accepted, before first-page delivery. Preserve failed Back
history, and await every parallel read before releasing pending guards. Use
partial projection DTOs only for controls, never as retained scientific specs.

Apply the same principle to stored candidates, not just drafts. The actual read
controller now uses a byte-bounded exact-revision preview with role counts and
explicit partial diagnostics; deliberate complete CLI reads remain available.
Follow returned cursors, commit navigation history only after successful reads,
and let Refresh retry the acknowledged cursor after a failed Back/Next. Clear
on selection/workspace/notified reload, fence late data/errors/continuations,
and disable runs while read/launch/metadata selection is pending. Detached old
row handlers must not launch. Explicit Run sends exact retained reference and
coordinates plus workspace identity, never a partial projection as a scientific
specification. Native capture/training guards remain authoritative. See
[candidate-browser.md](candidate-browser.md) for the tested concrete contract.

## Saved evidence and comparison recipes (Stage C domain extension)

Reuse the separation between inspecting immutable facts and explicitly requesting
execution. A structured view identifies its exact object revision and retained
attempt, never quietly advancing to a mutable head. A failed latest attempt must
not display an earlier success as its own result. Missing/unavailable values are
different from numerical zero and from a metric that is not applicable. Actual
execution counts and checkpoint identities are evidence, not decorative labels.
Summary rounding is presentation-only; expandable ledgers expose original values.
Tables keep numerical/status cells intact and scroll horizontally inside the
floating panel without changing base fonts. User metadata is literal text, not
HTML or implicitly interpreted math.

An ordered transient selection basket explicitly chooses immutable run references.
Adding the identical reference is a no-op; a different revision of the same
logical run requires explicit removal, not silently counting a retry as another
run. Create recipe is an explicit history-aware domain edit; file persistence is
a separate Save action. Viewing that recipe never executes it. Drilldown uses
the recipe's exact references, not current heads. Failures/cancellation/exclusion
remain visible. The view must not assert independent replication or aggregate
incompatible metrics merely because runs appear together.

Workspace change clears the transient basket and fences late replies. Origin
admission may already have happened; ignoring its late reply does not establish
detached persistence or crash recovery. A UUID check is not same-UUID/ABA safety.
Keep reads available during execution, with native mutations guarded through
pending terminal delivery, not only visually disabled buttons. Page bounds are
response/render limits, not a proof of bounded dependency/data validation.

The concrete implementation uses app/run_views.py, app/web/run-evidence.js,
app/web/evidence-view.js and app/web/science-results.js. Preserve their explicit
domains when adapting them; another application must define its own identities,
evidence grammar, units, history and statistical interpretation. See
[saved-evidence-views.md](saved-evidence-views.md) for the executable contract.

### Session recovery evidence and small terminal notifications

Do not automatically transfer full results/captured datasets after terminal
delivery, or repeat a complete recovery list on every state update. Keep full
evidence native and offer explicit read-only choice/tree pages and full export.
Recovery belongs to the window, not the active workspace or deleted worker.
Use a lifetime scope ID plus exact sequence for retained/exported marker changes;
label it session evidence, not a durable execution receipt. Preserve outside
pins, scientific value order, literal integer tokens and visibly partial previews.
Disable editing captured recovery values. Retry failed reads without losing Back
history; fence stale navigation/search continuations as well as their data replies.
Recheck scope/sequence/busy state after a file chooser returns, and refuse overwrite.
New browser projections do not change domain/SQLite/agent-import contracts.
Visual PyTorch's exact implementation is in [recovery-browser.md](recovery-browser.md).

### Exact retained artifact preview/export

An artifact action uses the displayed verified object/attempt, not unread input
text or a newer mutable head. Preview is literal source, scoped as generation
only; producing code is not executing it. Status/update counts travel with the
artifact so cancellation or an initial zero-update state cannot masquerade as
successful work. Scrolling long source inside the existing panel preserves user
font size rather than shrinking content to fit. Changing the displayed selection
clears old source and invalidates late preview replies.

External file creation is explicit and separate from domain history/file Save.
Capture verified exact artifact bytes before entering a native chooser's nested
event loop; recheck origin object/identity and busy state before creation. Cancel
does nothing. Refuse existing files/final symlinks rather than silently replacing
them. A file-write receipt identifies actual source hash and origin, not execution
or crash-safe recovery. Metadata/learned states may be sensitive even if record
payloads are omitted. In Visual PyTorch, app/run_export.py reuses the existing
inference lowering; see [saved-predictor-export.md](saved-predictor-export.md).

### Explicit asynchronous external work

The Notebook execution disclosure reuses the floating panel rather than adding
another native-window layout. Separate generation, input inspection and actual
execution: labels must describe which work can train or change external files.
An inspected selection pins exact input hashes; it does not stand in for a mutable
current object. Native confirmation defaults to Cancel, uses plain text and
preserves the full human app/action title. Put long technical identities in Show
Details instead of allowing them to dominate the dialog width.

Busy spans queued terminal delivery as well as the live worker. Inspection also
runs off the GUI thread. Stopping inspection has no execution receipt; stopping
real work retains actual partial evidence when cleanup succeeds, not a fabricated
success. Display replied/started steps separately from completed work. Reads stay
available, while edits and competing work have native admission guards.

Keep the lightweight GUI interpreter separate from the selected report Python.
Generation captures the exact dependency/data/runtime bytes before passing them
to that interpreter, then checks its returned recipe, mode, byte index and source
bytes before accepting the bundle. It creates a new directory without running
cells, launching a kernel or changing the workspace; it refuses overwrite.
Visual PyTorch's generation is synchronous with a 30-second process deadline,
not cancellable detached work. Inspection/execution use their existing threaded
adapters. Do not infer that generating a notebook needs Jupyter or Torch imports
in the GUI host, or that this separation makes trusted Python a security sandbox.

Readable receipt cards use theme tokens, base fonts, wrapping local paths/hashes
and a collapsed byte/hash ledger. Use literal DOM text, not trusted-looking HTML
from artifacts. Clear a previous verified card before a read that may fail due to
changed bytes. Workspace/request-generation fences prevent late inspection from
selecting a new origin. External artifacts and session history are not domain
undo or file Save; preserve those distinctions when adapting this pattern.
Visual PyTorch's exact actions/events and verified scope are specified in
[notebook-gui-execution.md](notebook-gui-execution.md), not a generic promise of
detached recovery or safe execution of arbitrary code.

### Scientific figure companions in exported reports

When the product exports numerical reports, make figure meaning recoverable
without hover or color alone. Preserve unlike quantities in separate plots,
explicit missing spans, distinct markers/line patterns, exact source identities,
checkpoint/population scope, source-unit versus transformed-loss meaning and
retention/discard information. Do not turn descriptive seed variation into
confidence intervals or hide failed trials to improve appearance.

Associate each figure with a readable caption and a bounded exact data disclosure;
state its full-data artifact and preview scope. Keep vector labels readable at
native size, allowing local scrolling in narrow readers. Scope description IDs
when multiple figures share a document. Measure actual rendered text rectangles
and inspect captures; character counts alone do not prove label containment.
Synthetic display stress values must be labelled as such, not passed off as fits.
These trusted-generated HTML report companions are not a reason to render arbitrary
artifact HTML in the editor's literal evidence cards. See Visual PyTorch's precise
[report-figure-presentation.md](report-figure-presentation.md) for the tested pattern
and font/frontend/accessibility/resource limits.
