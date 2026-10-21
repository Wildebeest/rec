# Sources

OpenCode v2.0.29 is MIT licensed; see LICENSE.opencode. `logo-slots.patch`
contains only the home-logo hook from pranavsreedhar123's
[logo-slot commit](https://github.com/pranavsreedhar123/opencode/commit/b70a0f8378a5faf928043a3d5ac5b17b5964ae98).
The upstream logo component stays unchanged. This patch is not merged upstream;
see [issue #51916](https://github.com/anomalyco/opencode/issues/51916).
`rec-home.patch` is Recursive's change: it roots OpenCode's data, config, state
and cache at `REC_OPENCODE_HOME` when that variable is set.

The four Recursive logo SVGs are the original assets used by the animated footer
at https://www.recursive.com/. They are Recursive branding, not part of OpenCode's
MIT license. Their source URLs, in layer order:

- https://cdn.prod.website-files.com/69c2d7e2ad6312d0a414043e/69fbe40d91b38edf2168e191_logo_frame_1_simple.svg
- https://cdn.prod.website-files.com/69c2d7e2ad6312d0a414043e/69fbe40d71fb648f7b27482a_logo_frame_2_simple.svg
- https://cdn.prod.website-files.com/69c2d7e2ad6312d0a414043e/69fbe40d5e56a1b574af9390_logo_frame_3_simple.svg
- https://cdn.prod.website-files.com/69c2d7e2ad6312d0a414043e/69fbdba8d07ce3f139737f3d_logo_frame_4.svg

Verified website timing on 2026-10-19: a nine-second linear opacity cycle,
0 → 1 → 0 at 0% / 25% / 50%, off until 100%. The three faces start at
0 / 2.97 / 5.94 seconds. The fourth layer is the persistent outline above them.
The terminal adaptation samples those polygons as stippled faces. Its nine-second
cosine light sweep keeps the whole mark visible, following the preferred terminal
mockup rather than reproducing the website’s opacity curve. Light and dark palettes
use lavender shades.
