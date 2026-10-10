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

## 2. Coordinate systems and stacking

| Layer | Coordinates | Persisted in Visual PyTorch | Movement |
| --- | --- | --- | --- |
| Canvas world | Model/world units | Box positions and graph view | Free block drag; Ctrl snaps; empty drag pans |
| Floating menus | CSS viewport units relative to editing stage | No; per-view session | Menu drag; Shift-drag translates menu layer |
| Whole interface | Qt WebEngine zoom | App config JSON | Ctrl+/Ctrl−; not canvas zoom |

Canvas transforms are translate(view.x,view.y) followed by scale(view.zoom).
Convert pointer positions through the inverse transform before moving a box.
Snap grid is 24 world units for boxes, 24 CSS stage units for menus. Grid origin
is (0,0) in each layer, not the window decoration's top-left. UI zoom scales page
appearance, but does not replace these logical coordinate definitions.

The editing stage is positioned, isolated, clipped to its rectangle. Menus are
absolute descendants, not OS top-level windows. They therefore cannot draw over
window chrome or outside the application's actual surface. "Off-window" means
most of the rectangle can be outside that clipped stage; recovery uses its slice
that is still visible. Native detached palette windows are NOT implemented.

Normal menus start at z-index 2; interacting/opening raises them through a monotonic
counter. Expanded menus are above ordinary menus. The pack is assigned a very high
z-index so it remains accessible. This is a practical prototype stacking scheme,
not a general compositor. Popups/modal dialogs use their browser/native mechanisms.

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

- Ctrl + menu/pack drag: snap its **top-left** point to nearest 24-unit grid point.
- Ctrl + resize: snap **bottom-right**, preserving top-left where feasible.
  Choose a grid-aligned corner satisfying minimum size.
- Ctrl + Shift layer-drag: snap pack top-left as the shared anchor; preserve
  relative positions by applying the same delta to ordinary menus, including hidden ones.
- Visibility constraints can override exact snapping at stage extremes. Expanded
  menus do not move/resize and are excluded from layer translation until restored.

On stage resize, refit expanded menus and constrain moved ordinary rectangles/pack.
Closed menus need no immediate painted layout; when measuring one, briefly remove
hidden synchronously and restore it before the next paint. Never interpret a
display:none zero rectangle as its stored geometry.

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
  If usable space is absent/small, request that the user hide/move a menu.

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
Compact width is 72 world units; height is at least 54 for Linear/Conv2d or 44 otherwise,
increasing to keep multiple ports distinct. Small 9-unit monospace text shows
input/output shape, Linear last-axis in→out widths, activation name or type-specific
summary. Linear/Conv2d also have an 8-unit Bias on/off line. Summaries truncate after
13 characters with full hover text; missing inferred dimensions show ?. The type
icon sits at the compact body's top-center and remains directly draggable.
Folding preserves preferred expanded sizes, tensor
semantics, datasets and results. Its boolean collapsed flag persists with the
block and participates in undo/redo. Training disables folding and movement.
Bézier bend is at least 50
units, otherwise approximately 45% of horizontal separation. View wheel zoom uses
the pointer as anchor; graph zoom is clamped to 0.05–2.5; Fit caps initial scale at 1.3.
These numbers are defaults, not domain laws. Menu CSS and diagram SVG text must
remain independently tuneable when adopting larger readable defaults.

Every box has a 14×14 bottom-right SVG resize target with two diagonal marks,
nwse-resize cursor, accessible label/disabled state and keyboard focus. Drag in
inverse canvas/world coordinates; do not change top-left. Ctrl snaps bottom-right
to the 24-unit world grid, raising a snapped corner as necessary to satisfy the
minimum. Arrow keys change a focused grip by 8 units; Shift or Ctrl uses 24, and
Ctrl also snaps. Inspector Width/Height is an alternate input; Automatic size
deletes both preferences. Do not start inspection on grip double-click.
Capture the persistent canvas, not the re-created SVG node, during drag. Redraw
nodes and wire endpoints together. Release/cancel/lost-capture/blur ends the
gesture, clears resize CSS state and persists one edit if moved. Training disables
resizing. Saved sizes participate in workspace undo/redo, but preserve code,
contracts, datasets and learned results because they are presentation-only.

Single click selects but need not open a hidden Inspector; double-click opens it.

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
port identity or compute order. Current control values apply on each button press;
Save layout preferences explicitly persists both profiles at user level. Settings
do not live in .vtw and do not create workspace undo entries.

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
folding, positions and view; expanded width/height preferences are retained.
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
contextAt returns null outside a box, otherwise {id,modelId,canDelete} after
selecting it. On a box the native menu begins Inspect block, then Delete block,
then a one-pixel horizontal separator. General graph/panel actions follow;
a second separator precedes workspace actions. General groups remain alphabetical;
the block section is intentionally ordered by inspection before destruction.
Both block actions capture the exact IDs. Inspect dispatches inspect-block,
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
Activation/Reshape, Concat, Splitter, Replicator, reference interfaces, training
contracts, MSE, constrained runs. Reference normal double-click inspects; Ctrl/
Meta-double-click opens the referred graph in a new tab. Inspector also has Open
details/Open in new tab actions. Socket click wiring and arbitrary DAG checks live
in the domain adapter, not in the floating-menu manager.

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
Respect reduced motion. Use a separate message span so updates retain the button.

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

I/A/B/E toggle Inspector/Architectures/Building blocks/Experiment, ignoring
editable fields, modified/repeated keys and modal dialogs. Ctrl+Z undoes workspace
edits; Ctrl+Shift+Z/Ctrl+Y redo; text fields retain native text undo. Native Edit
actions dispatch the same commands without stealing text-field shortcuts.
Domain history is session-only SQLite snapshots (50 steps/128 MiB combined),
not floating-menu/display preferences. No history edits during training;
preserve run evidence for surviving restored architectures. A different product
must choose its own transactional history implementation, not blindly reuse .vtw.

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
