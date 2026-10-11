# Visual PyTorch contributor and agent instructions

This repository is the canonical application source for
https://github.com/phao/visual-pytorch. Keep source, tests, portable specifications,
examples and release documentation here. Private campaign records and research
workspaces are not part of the public product. Do not create nested repositories
or maintain another live application copy in a research campaign.

## Scratch, campaign evidence and source ownership

When sole product development is active, work in this visible canonical checkout
on the current task branch. Separate worktrees remain useful for real concurrent
writers, but do not create hidden routine source forks under `.cpw-workspace/`.
The public product is shared across campaigns, like shared project context;
campaign source snapshots are historical, not another live application.

Use Git-ignored `.scratch/` for disposable test fixtures, temporary profiles and
environments. Set TMPDIR to that existing directory for bounded tests when useful.
Keep meaningful private research/test evidence and handoffs in the owning campaign
under its write lock, not in the public product or CPW bookkeeping directory.
Reserve `.cpw-workspace/` for CPW-owned locks, inboxes, activity/provenance state and
their receipts. Older evidence/environment paths remain valid until explicitly
and safely reorganized; never broadly delete them or break pinned references.

## Git and collaboration

Use branch → work → validate → commit → integrate, targeting `main`. Use the
`git-workflow` skill when installed; these rules also apply without that skill.
Use separate worktrees/indexes for concurrent agents. Preserve user changes.
Serialize integration with an advisory `flock` on `agent-integration.lock` inside
the absolute directory from `git rev-parse --path-format=absolute --git-common-dir`.
Hold it across target-tip/ownership/cleanliness checks, merge and verification
in the same process. A private campaign lock does not protect this repository.
Never force-push, rewrite shared history or publish secrets. Credentials must
remain in the local credential manager/SSH agent, never chat, source or logs.
Remote publication needs the applicable user authorization and authentication.

Record meaningful decisions, checks and remaining work in commit messages and
`docs/development.md`; do not imply a roadmap stage is complete from an import.

## Product invariants

Preserve the compact floating-panel Qt/WebEngine UI, keyboard/typing safeguards,
themes/preview, separate menu/box spacing, readable fonts and undo/redo. Compact
block icons must remain draggable, with ports usable and computation unchanged.
Keep `.vtw`, CLI and reusable GUI specifications aligned with executable code.
Sort unordered browsing lists alphabetically; never sort tensor packing, ordered
ports/Concat sources, edit batches or recent-file history.

Research primary documentation and literature when useful; record significant
references and distinguish verified behavior from inspiration. Do not edit tests
or evidence simply to hide a defect. Run bounded fits with conservative resources.
Keep Chromium sandboxing enabled. Never restart a user's app or alter a user's
workspace/configuration for tests. Use temporary private workspaces/profiles.

## Public presentation and licensing

Describe this as a research prototype, openly AI-coded and jointly human/AI
directed. Use real app captures for screenshots; label mockups and future features.
Original project code/documentation are GPL-3.0-only. Preserve third-party
licenses and notices; do not relabel dependencies. Do not promise that licensing
eliminates all liability or that all generated helper code is license-exempt.
