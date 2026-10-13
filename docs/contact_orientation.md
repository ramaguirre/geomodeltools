# Contact orientation from outcrop traces (rule of V's)

`geomodeltools.contacts`.

## Idea
A planar contact crossing relief draws V's in the valleys: the strike follows equal elevation, the V points down-dip, and the dip
follows from the elevation range against the horizontal offset (the exception: dip in the stream direction and gentler than the
stream gradient, where the V points upstream). A plane fitted in 3D to the trace (x, y, z from a DEM) contains all of that, so the
fit needs no separate V rule: `fit_contact_plane` returns strike, dip, dip direction, residual, relief and the standard error of the
dip.

## When the answer is refused (`ok` False, reasons listed)
fewer than 12 points; relief below 100 m (60 m for local windows); residual above 6% of the relief (not a plane: fold, pluton margin,
fault, cover); standard error of the dip above 5 degrees (trace too straight or short); dip above 75 degrees or below 3 degrees.
Also do not use it for: curved intrusive margins, faulted contacts and unconformities, cover deposits, contacts the mapper drew
by interpretation or under cover, dip close to the stream gradient, DEM error comparable to the relief.

## Use
`contact_points(lines, dem, spacing)` densifies and drapes shapely lines; `pooled_fit` fits several traces as one surface; window the
trace for local readings and keep the accepted windows as structural data. Tests use synthetic planes of known dip and direction.
