# Visual PyTorch floating-menu interaction specification

The app is a PyQt6 WebEngine desktop application with one HTML/SVG editor.
The portable design contract for reuse in other projects is
[UI-AGENT-HANDOFF.md](UI-AGENT-HANDOFF.md); this document records app-specific bindings.

## Menus and independent layers

Architectures, Building blocks, Experiment, Inspector and Layout start hidden. One floating
vertical PANELS pack independently opens/hides each. Click its icon to hide all
menus and shrink the pack. Double-click the compact icon, or focused Enter/Space,
restores only the menu buttons. Palettes are arbitrary rectangles, not fixed sidebars.
The pack starts flush at editing-stage (0,0), below the header, and Reset returns
it there; it is still draggable. For spacing u, inter-button gap is u, side/bottom
list margins are 2u (both doubled), and top margin is u. Button interiors retain
0.75u padding; readable fonts do not shrink.

× hides; ⤢ expands/restores. Titles, frames and empty content areas drag menus;
text, form controls, code blocks and scrollbar regions keep normal interactions.
Menus may leave the stage, retaining a 24-unit visible slice. Reset restores
initial geometry and closes menus without destroying their contents.

Normal empty-canvas drag pans the graph. Shift + primary-button drag on stage
translates the menu layer, including hidden ordinary menu positions, excluding
full-stage expanded menus and text-input interactions. Ctrl-drag snaps a menu's
top-left to the 24-unit stage grid; Ctrl-resize snaps bottom-right, preserving
top-left/minimum size where feasible. Ctrl+Shift snaps the pack anchor and applies
one shared delta. Normal box Ctrl-drag uses the separate 24-unit world grid.

Menus have overlay scrollbars, with hover/focus/scroll reveal and drag/keyboard
control. Architectures is no longer nested in Building blocks. Its
model list has natural height and no inner scrollbar or collapsible subsection;
the outer menu
scrolls if necessary. Building blocks retains the separate reusable-architecture
selector. No Create MLP button/dialog is present.
Architectures' title owns +; its non-collapsible Currently open card precedes
the list and shows name/counts/shape/parameter information or draft status.
Building-block buttons show icons/names only, with descriptive hover tooltips
and accessible labels. NN boxes have distinct top-right vector type icons.
Related selector/action pairs have a small rounded, one-pixel theme-token frame:
Nonlinear activation + Activation, and Reusable architecture + Architecture
reference. Each is a labelled semantic group and stays together when browsing
entries are alphabetically sorted. Standalone actions remain unframed. Internal
padding/gap use the existing menu-spacing unit; child margins are removed so
the frame does not reintroduce oversized whitespace. This is a visual grouping,
not an additional collapsible menu or graph/computational edit.

The architecture toolbar occupies the center of the top header. Canvas geometry
does not shrink when a menu opens. Fit finds the largest unobscured rectangle;
hide/move menus if no useful area remains.

## Diagram and accessibility behavior

Single click selects; double-click any box opens its Inspector. Normal reference
double-click inspects; Ctrl/Meta-double-click opens its internals in a new tab.
Open details/new tab also remain Inspector commands.

Inputs/Outputs declare sample tensor shapes; Reshape preserves B; internal
Splitters and independent input branches compile as a port-aware DAG. Socket
wiring and precise source-port wire deletion match the file-format contract.
The canvas has single box/wire selection (group selection remains deferred).
Unmodified D deletes the selection; Delete/Backspace do so with canvas focus.
Text controls/contenteditable, modal dialogs, Ctrl/Alt/Meta-modified or repeated keys, active drag and
training guard deletion. Existing minimum-two-block and unary-bypass rules apply.
Each incoming arrow has a small draggable handle 11 world units before its socket,
drawn above boxes. Selecting a wire highlights it and enlarges the handle.
Dragging that handle previews a replacement destination but leaves the graph
untouched until pointerup on an empty input socket. A Concat's final empty socket
appends the source/port key, and departure removes its old ordered key only.
Source ID/from_port and sibling wires survive. Occupied sockets, absent endpoints,
self-links/cycles, blank releases, Escape, pointer cancellation/lost capture and
window blur cancel without a graph/history edit. Successful release is one
undoable computational edit and invalidates current learned results. Tensor
shape errors may still create a saveable non-runnable draft; no shape is guessed.
Every diagram block has a bottom-right resize grip: Ctrl snaps that corner to the
24-unit world grid, preserving top-left and readable minimums. Focused grips use
8-unit arrow steps, or 24 with Shift/Ctrl. The block context menu's Automatic size
removes custom dimensions; Inspector exposes no dimension/removal controls.
Sizes persist in .vtw, follow undo/redo, update wires/
Fit and never change tensor shapes, datasets or learned weights. Linear footers
always start Bias on/off; counts and weight dimensions use only the last axis.

Default font is 14px with UI zoom 1.15. Options → Display preferences edits font/
zoom/theme, numeric 1–10 menu spacing (default 4) and separate 1–100 NN box spacing
(default 35). Menu content uses that value as a CSS-pixel padding unit without
scaling fonts or frame/heading drag handles. Legacy Compact/Spaced preferences
load as 4/10 without writing; explicit Save persists the numeric value.
Defaults: Dark, menu spacing 4, box level 35. Preview updates live without closing or
writing preferences; Cancel restores; Save persists. Intrinsic pack size and
font-dependent panel minima prevent launcher clipping. Active drags suppress
selection/show a grabbing hand, while idle panel text remains selectable.
Ctrl+/Ctrl− changes page UI zoom across tabs and persists to
~/.tensorloon/config.json, separately from saved diagram pan/zoom.

Application confirmations use HTML task headings with Cancel focus, never a
file-origin/path title. Native Qt custom context menus offer app actions only,
not browser Save page/View page source/Reload. Ordinary left-click remains unchanged.
Right-click selects before showing that menu; coordinates account for page zoom.
On a box, the first section is Inspect block → Delete block → thin separator.
Inspect selects the exact clicked block and opens its Inspector; it is read-only
and remains available during training. This deliberate action order takes
precedence over alphabetical browsing order. The general graph/panel group and
the final workspace group remain separately alphabetical, with a separator.
Delete block targets that exact block/architecture and uses workspace
undo/redo. It is disabled during training or at the two-block grammar minimum;
empty space has no delete action. Inspector has no Delete architecture button.
Native menu bar/dropdowns/tabs follow all 17 theme choices, including Metallic.
Themes, architecture/reference names, activations and unordered block/launcher
choices are alphabetical within logical groups, without changing defaults.
Recents and computational/port/packing/edit sequences retain meaningful order.
Metallic adds steel/gunmetal web gradients. Math favours serif prose/labels/titles
in web and Qt, retaining monospace code/shapes. Leaving Math or canceling Preview
restores the baseline Qt family. No fonts are downloaded.
I/A/B/E toggle panels except while typing, using modified keys or in dialogs.
Empty-canvas clicking/panning keeps selection. Escape cancels active wiring/drag
first; otherwise it clears selection, or hides panels and shrinks the pack when
nothing is selected. It does not reset panel geometry. Typing/dialogs keep Escape.
Workspace Ctrl+Z/Shift+Z/Y history does not capture floating menus/display settings.
Validation banners are upper-right. Bottom notifications last 5.5s, fade 0.25s
in/out and have × dismissal, respecting reduced-motion preferences.
Offline MathJax TeX/AMS SVG rendering is enabled for opt-in math containers and
Inspector formulas; see the portable handoff for API, provenance and safety limits.

## References and verification boundaries

[Qt tool palettes](https://doc.qt.io/qt-6/qdockwidget.html) inform movable/closable/
expandable controls; actual palettes are HTML rectangles, NOT QDockWidgets.
The shared pack and layer/grid gestures are the user's design.

[Qt context-menu policy](https://doc.qt.io/qt-6/qwidget.html#contextMenuPolicy-prop)
supports the app's CustomContextMenu signal/assembly.
[Qt confirmation API](https://doc.qt.io/archives/qt-6.9/qwebenginepage.html#javaScriptConfirm)
explains the browser-confirm mechanism avoided by titled application HTML dialogs.

Source checks and a tiny DOM harness cover menu geometry/state/gesture contracts,
while a bounded real WebEngine follow-up now checks launcher pixel geometry,
font stacks, alphabetic browsing, bias state, real Qt-mouse block resizing and
context hit-testing/editor deletion/undo/redo via a private
workspace. Native Qt tests cover menu assembly, display persistence and tab style.
Desktop readability/dialogs/full interaction and Alt-Tab/Dolphin integration
still require a normal-desktop trial. Chromium sandboxing was retained.
