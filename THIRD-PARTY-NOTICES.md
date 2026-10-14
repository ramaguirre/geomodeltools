# Third-party notices — geomodeltools

`src/geomodeltools/static/plotly.min.js` is a vendored copy of Plotly.js, inlined into every
report `write_html()` writes (see `report.py`). Its licence allows commercial use and
redistribution, provided the notice below travels with the file. This copy, and this notice,
were taken from the `swath_local` repo's own vendoring of the same library.

geomodeltools's own code (`src/`, `tests/`, `notebooks/`) is not covered by this file; it is
MIT-licensed (see `pyproject.toml`).

| Component | Version | Licence | Where from |
|---|---|---|---|
| Plotly.js (charts) | 4.1.1 | MIT, © 2012-2026 Plotly, Inc. | `src/geomodeltools/static/plotly.min.js` |
| Plotly's own dependencies | see below | MIT, ISC, BSD-3-Clause, Zlib | bundled inside Plotly |

No copyleft (GPL/LGPL/AGPL) and no non-commercial licence is involved.

## MIT licence text (applies to Plotly.js and every MIT-licensed component below)

Copyright holders: Plotly.js — © 2012-2026 Plotly, Inc.; and the holders listed below.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Components bundled inside Plotly.js, MIT-licensed (each © its author, same MIT text as above)

- @plotly/d3-sankey-circular — Tom Shanley, 2017
- @plotly/point-cluster, regl-error2d, regl-line2d, regl-scatter2d, regl-splom, gl-text, has-passive-events, svg-path-sdf — Dmitry Ivanov, 2017-2018
- @plotly/regl, strongly-connected-components — Mikola Lysenko, 2013-2016
- @turf/area, @turf/centroid, @turf/meta — TurfJS, 2017
- base64-arraybuffer — Niklas von Hertzen, 2012
- country-iso-search — Plotly Technologies Inc., 2026
- culori — Dan Burzo, 2018
- fast-isnumeric — Alex Johnson, 2015
- mouse-event-offset, webgl-context — Matt DesLauriers, 2014
- parse-svg-path — Jake Rosoman, 2013
- point-in-polygon — James Halliday, 2016
- polybooljs — Sean Connelly, 2016
- probe-image-size — Vitaly Puzrin, 2015
- world-calendars — alexcjohnson, 2016
- is-mobile, has-hover, native-promise-only — their respective authors
- is-plain-object (transitive) — Jon Schlinkert, 2014-2015

## BSD-3-Clause components bundled inside Plotly.js

- @plotly/d3 — Copyright (c) 2010-2016, Michael Bostock
- @plotly/d3-sankey — Copyright 2015, Mike Bostock
- maplibre-gl — Copyright (c) 2023, MapLibre contributors

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## ISC components bundled inside Plotly.js

- d3-force, d3-geo, d3-geo-projection, d3-hierarchy, d3-interpolate, d3-time, d3-time-format — Copyright 2010-2024 Mike Bostock
- d3-format — Copyright 2010-2026 Mike Bostock
- topojson-client — Copyright 2012-2019 Michael Bostock

Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is hereby granted, provided that the above copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

## Zlib component bundled inside Plotly.js

- gl-mat4 — Copyright (c) 2013 Brandon Jones, Colin MacKenzie IV

This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

## Scope of this list

Taken from Plotly.js 4.1.1's published dependency list (npm registry) with each package's licence
file; the copyright lines are those of the latest published version of each package. It covers
Plotly's direct dependencies. Packages those pull in in turn (for example maplibre-gl's own
dependencies) are not individually listed. Re-check this file whenever
`src/geomodeltools/static/plotly.min.js` is replaced.
