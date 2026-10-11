# Workspace-owned floating panel arrangement

Layout → Floating panel arrangement → **Save current arrangement** records an
explicit preset in the current workspace draft. Save workspace writes it to `.vtw`.
**Restore saved arrangement** reapplies it. Opening the workspace or an architecture
in a new tab initializes from this preset; ordinary moves remain tab-local and do
not autosave or create undo steps. Existing tabs receive preset metadata but do
not jump when another tab saves. Reset floating menus is a local reset, not a
deletion of the saved preset. CLI `panel-layout` with `value: null` clears it.

Opening from the Architectures list hides all menus and collapses the pack,
clearing the selection like two Esc presses, after successful navigation. This
applies to same-tab and new-tab opens; the latter suppresses restored visibility
once on initial load. It does not rewrite the saved preset, and other navigation
does not inherit this menu-specific behavior. Failed opens preserve the menus.

The PANELS launcher and all palettes share click-to-front stacking above the
network, including expanded palettes. Clicking the exposed launcher can raise
it above a palette; clicking or opening that palette can raise it above the
launcher again. Restoring expanded geometry does not undo recent stacking.
The version-1 preset stores relative palette order only, not the launcher's
transient position in that order; applying it starts the launcher at the base
menu level. Reset floating menus recovers a covered launcher. Building Blocks
no longer contains the former “A LITTLE GUIDANCE” section; per-block About
information and tooltips are unchanged.

## SQLite and CLI contract

Optional `metadata` key `panel_layout` contains JSON with this closed version-1
shape (no new tables or container version):

```json
{
  "version": 1,
  "pack": {"x": 0, "y": 0, "collapsed": false},
  "panels": {
    "architectures-panel": {"x": 180, "y": 10, "width": 320, "height": 450, "visible": false, "expanded": false},
    "experiment-dock": {"x": 40, "y": 400, "width": 700, "height": 300, "visible": false, "expanded": false},
    "inspector-panel": {"x": 700, "y": 10, "width": 320, "height": 450, "visible": false, "expanded": false},
    "layout-panel": {"x": 220, "y": 10, "width": 340, "height": 500, "visible": false, "expanded": false},
    "library-panel": {"x": 180, "y": 10, "width": 300, "height": 500, "visible": true, "expanded": false}
  },
  "order": ["architectures-panel", "experiment-dock", "inspector-panel", "layout-panel", "library-panel"]
}
```

Coordinates are stage-relative CSS pixels, not model/world units. They must be
finite numbers in ±1,000,000. Ordinary widths/heights must be finite numbers in
[1, 100,000]. Version must be integer 1; flags must be JSON booleans. All five
panel IDs and fields are required, with no unknown fields. `order` is a unique
permutation from bottom to top; never alphabetically sort this semantic order.
Hidden panels cannot be expanded; a collapsed pack cannot have visible panels.
Expanded panels store their ordinary restore rectangle plus `expanded: true`,
not the full-stage rectangle. No CSS text is persisted.

The renderer enforces current typography-dependent minima and a recoverable
24-CSS-pixel slice on both axes. Small windows or different fonts can therefore
adapt the rendered rectangle without changing the stored preset. Exact grid
alignment can yield to visibility/minimum-size constraints. Missing metadata
means the legacy defaults. Invalid stored metadata fails opening rather than
silently guessing; opening a legacy workspace does not write its source file.

CLI `inspect` includes `panel_layout`. Within an ordered atomic `apply` batch:
`{"op":"panel-layout","value": <preset-or-null>}` uses the shared validator.
Normal file revision conflict checks and atomic save behavior apply. No-op
presets do not mark dirty. Computational undo/redo preserves the *current*
panel preset, since it is presentation state outside that history.

Native `setPanelLayout` carries `workspace_id`, ephemeral `panel_scope`,
`expected` preset and `layout`. The scope changes even when reloading the same
workspace UUID; compare-and-set rejects stale tab writes. The acknowledgement
and `panelLayout` broadcast contain scope/identity/preset/dirty only, with no
whole-model/data reread. Presentation saves are allowed during training; this
does not broaden authority for computational edits. The scope is not persisted.

## User-owned snap preference

Options → Display preferences has independent **Snap blocks to grid by default**
(`grid_snap`) and **Snap menus to grid by default** (`menu_grid_snap`) strict
booleans in user config (both default false). Preview is transient; Cancel
restores the original value; Save persists it. Legacy config reads remain
nonmutating. This interaction default is global, unlike the workspace preset.

For each movement/resize event: snapping = that layer's default XOR Ctrl. With the default
off, Ctrl enables snapping; with it on, Ctrl permits free movement. Block drags
snap top-left in the 24-world-unit grid; block resizes snap bottom-right. Panel
and pack drags snap top-left in the 24-CSS-pixel stage grid; panel resizes snap
bottom-right. Shift-drag still moves the menu layer; it uses the pack as a shared
snap anchor and preserves relative panel offsets where bounds permit. Ctrl
inverts the menu default for that gesture too, independently of the block option.
Focused block-grip arrow resizing uses the block rule;
Shift/Ctrl still choose its larger step. Group selection remains deferred.

The plain page Find toolbar/menu/shortcut command is removed. This is not an
architecture search implementation; a replacement remains future work.

## References and verification boundary

Stage rectangles are measured using the browser's
[CSSOM View `getBoundingClientRect`](https://www.w3.org/TR/cssom-view/#dom-element-getboundingclientrect).
This explains the CSS-pixel versus model/world distinction; executable tests
are the authority for the app behavior, not a claim that arbitrary desktop
zoom/layout configurations have been certified. Pure JS tests cover preset
validation, gesture cleanup and snap inversion. Python/CLI tests cover strict
storage, legacy reads and undo preservation. Native and actual sandboxed
WebEngine checks exercise private workspaces/profiles; user files and settings
are never altered by these tests.
