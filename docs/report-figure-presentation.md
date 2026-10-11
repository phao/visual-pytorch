# Rendered report figures — local Stage C checkpoint

This is the contract for `app.report_runtime.curve_svg`, `curve_figure` and the
generated notebook's training section. It changes presentation, not `.vtw` v6,
ReportRecipe syntax, stored training metrics, split identities or optimizer behavior.
The complete Stage C exit audit and D/E remain separate work.

## Figure question and numerical meaning

Each figure asks how one named head's weighted population loss changed over the
retained successful optimizer-update checkpoints of one exact run. Train and
validation remain separate populations. They are not the combined weighted
multi-head objective, independent experimental replicates or a general ranking
of architectures. Unlike heads/datasets are never pooled on one axis.

Values come directly from retained head metrics; this presentation layer does not
recompute losses, refit transforms, train, smooth or invent uncertainty. Both axes
are linear, including genuine zero values. Missing values break lines. Train uses
solid lines/circles; validation uses dashed lines/squares, with explicit labels.
Final test metrics are not learning curves and are never plotted here.

Captions identify the run/revision, final checkpoint, dataset, source target unit,
ordered target transforms, retained/observed/discarded evaluation counts and the
conditional population meaning. The source unit is not asserted to be the unit
of a standardized loss; MSE, MAE and cross-entropy retain their own definitions.
Every retained evaluation identifies its checkpoint, effective denominator and
evaluated-record count in the companion CSV. Interpolation across missing spans,
confidence intervals and statistical independence are not implied.

## SVG and notebook presentation

The numerical SVG metadata remains version 1 with exactly its existing run/head,
checkpoint/context/series/domains/linear-scale/retention fields. Numerical series
and axis domains are unchanged; absolute SVG positions reflect the new layout
offset when a title wraps onto extra lines. At native width 760, axis
and caption text is 14 pixels, title text 18 pixels. Monospaced titles/captions use
explicit wrapping, treating East Asian wide/full-width characters as two cells
and combining marks as zero cells. This is a layout heuristic verified in the
test host, not a font-shaping guarantee for every script/viewer. Long labels remain
complete in metadata and the accessible description; they are not ellipsized.
First/last x-axis tick labels anchor inward to preserve long update counts.

SVG title/description IDs are deterministic scoped hashes of exact run/head/context,
so different plotted identities can coexist inline without referring to another
figure's description. Repeating the identical artifact twice in the same DOM still
requires the embedding consumer to manage duplicate instances; hash identity is
not a universal DOM-instance allocator.

Newly generated training cells display a `text/html` figure companion rather than
an isolated SVG MIME output. The same original SVG remains a standalone retained
artifact. `curve_figure(path)` wraps it in `figure.vpw-curve` with a scoped
`figcaption`, literal escaped title and full description, and a collapsed native
details/summary for exact plotted rows. Wide plot/table content scrolls locally;
the SVG is not shrunk into unreadable type in a narrow notebook column. HTML
captions inherit the notebook reader's font size. Column headers have scope=col;
table cells explicitly keep whole words/numbers and override caption wrapping,
with local horizontal scrolling rather than broken digit/word fragments.

The preview shows the first at most 100 rows, states the total retained CSV row
count, and names the complete CSV in the same attempt directory. It preserves
column order: update, checkpoint, population, head, kind, value, denominator,
records. An empty value is labelled unavailable, never zero. Other strings/numbers
are displayed exactly as read. A missing CSV, wrong header or incomplete/extra-field
row raises an error rather than fabricating companion data. The full CSV is not
truncated by this display bound. Reading/counting the file is still proportional
to its size; this is not a general transport/storage/validation resource bound.

The helper assumes the application's just-generated SVG and CSV. It escapes
descriptions/CSV cells but embeds the generated SVG; it is not a sanitizer or
integrity validator for arbitrary user-edited SVG. The trusted notebook already
executes Python. Do not route this helper's output into the editor's arbitrary
artifact viewer: that GUI continues displaying literal revalidated receipt cards.
Captured bundle/terminal integrity checks remain responsible for file hashes.

No new numerical or authoring dependency is introduced by the helper. It uses
stdlib XML/CSV/Unicode handling. New notebook source and runtime bytes have new
captured hashes; old complete bundles retain their own pinned runtime/source and
do not silently change appearance or scientific identity. No SQL schema or CLI
command changes are required.

## Reproduction and actual delivery checks

Eight pure tests in `test_report_figures.py` check distinct identity IDs, wide/
combining text without truncation, unchanged numerical series/domains/missingness,
literal semantic captions and CSV values, a 100-of-240 display bound, empty/invalid
CSV behavior and a fresh stdlib-only host with no numerical/Qt/Jupyter imports.
Existing real-kernel tests also verify exact SVG coordinates and CSV/history
identity, portability, saved-result no-training and explicit rerun behavior.

`report_render_fixture.py` explicitly executes a saved fit, retains a missing-worker
failure, and produces/executes both complete notebook modes through actual kernels.
The rerun is two models × two datasets × two declared initializations: eight fresh
fits. All nine report sections and original SVG/CSV artifacts are preserved.
The fixture independently checks each SVG metadata series against the corresponding
CSV update/value rows, with explicit checkpoint/head/denominator/record fields.
`nbconvert`'s basic HTML exporter is used only for the acceptance fixture, with a
small explicit local stylesheet; it is not an application runtime dependency or
a new product HTML export command. Install `requirements-test-report-render.txt`
in the numerical test environment for this optional check.

For example, from the product checkout with the existing numerical and Qt test
environments:

```bash
report_render_tmp="$(mktemp -d)"
PYTHONPATH=.:tests OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /path/to/numerical/python tests/report_render_fixture.py "$report_render_tmp"
PYTHONPATH=. QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu \
  VTW_REPORT_CAPTURE=1 /path/to/qt/python tests/qt_report_render_smoke.py "$report_render_tmp"
```

The real sandboxed off-record WebEngine check renders those executed notebook HTML
documents with network requests rejected. It checks all sections and actual saved/
rerun statuses, escaped literal content, semantic figure/caption relationships,
keyboard expansion, inherited caption font size 28 and local overflow at width
520. A separate gallery checks
the nine actual SVGs and two explicitly hand-declared synthetic layout stress
cases (long W strings, wide Unicode labels and large update counts). Synthetic
stress values are not training observations or research evidence. Browser text
rectangles check clipping and label overlap; exact IDs check inline description
collisions. Actual saved/rerun/long-label and grayscale captures require visual
review as well; grayscale is a diagnostic, not an accessibility certification.

The original host reproduction found twelve overflowing text rectangles and
twenty repeated title/description IDs. After repair those checks have no clipping,
overlap or duplicate IDs in the eleven-figure gallery. This does not certify all
browser fonts, Jupyter frontends, print/PDF renderers, arbitrary labels, screen
readers or WCAG compliance. The Qt reader imports no numerical/Jupyter stack.
Full report/table/metadata/data/manifest scalability, richer in-app figures and
the complete C acceptance review remain open.

## Design references

The scientific-plotting skill guided checking delivered-size labels, retaining
missingness, redundant population encodings, exact companion data and scoped
scientific captions. Primary references include
[W3C complex-image descriptions](https://www.w3.org/WAI/tutorials/images/complex/),
[the nbformat MIME/metadata contract](https://nbformat.readthedocs.io/en/latest/format_description.html)
and [nbconvert's HTML/basic output](https://nbconvert.readthedocs.io/en/latest/usage.html).
The installed nbconvert 7.17.1 basic/classic SVG template encodes an isolated SVG
as an image; the new companion preserves description/data in ordinary adjacent
HTML instead of relying on that image's internal SVG title alone. This is our
tested presentation design, not a guarantee that every notebook renderer conveys
identical accessibility semantics.
