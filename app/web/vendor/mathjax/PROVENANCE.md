# Vendored offline math renderer

MathJax 2.7.9, copied unmodified from the installed Debian libjs-mathjax runtime
under /usr/share/javascript/mathjax on 2026-10-10. Upstream:
https://github.com/mathjax/MathJax/tree/2.7.9 . Apache-2.0 license and Debian's
full copyright/font attribution are retained in LICENSE and DEBIAN-COPYRIGHT.

This is a selected TeX/AMS/SVG runtime closure (2.6 MiB), including the Safe and
AssistiveMML extensions, TeX SVG glyph data and English localization. It excludes
other output engines, other font families and input syntaxes. It is not a claim
to ship the entire upstream distribution or the latest MathJax release.
The installed, available legacy bundle was used because shell network/DNS access
prevented obtaining a newer release. The app adapter isolates the legacy API for
a future audited upgrade; no CDN or local operating-system path is needed at run time.

SHA-256 pins:

- MathJax.js: 0d588838c61dc2533f6b1aa81833de5327f4bab2e81cc3784000812b2079f14c
- config/TeX-AMS_SVG-full.js: 8b4120932f298a1f18fa3f39c80e5d9e1991aec9349086335d8d20245b824158

Do not edit vendor files. Update from a reviewed distribution, retain licenses,
record provenance and rerun adapter plus normal-desktop rendering checks.
