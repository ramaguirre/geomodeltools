# Overburden basins: building the base surface of a sequence of basins outside the modelling software

Audience: a geologist who has to model thin, widespread cover (alluvium, colluvium, moraines, landslide deposits) in
Leapfrog Geo, **and** a smaller local language model that is learning to do the job step by step. Every step says what
goes in, what comes out and how to tell that it worked. No real project data appears here; the examples use "basin A,
basin B, basin C".

## 1. What problem this solves

A geological map shows cover units as polygons. Their thickness is unknown. Tools that build a surface from points on
the ground (a contact surface "from points", or guide points) work badly for thin cover: the polygon outline and the
topography are almost the same surface, so the cover comes out with no volume or with a ragged volume.

The approach here builds the **base of each basin** (the floor of the cover) as an ordinary triangulated surface in
Python, using only the polygon and the DEM. The surface is then imported into Leapfrog and used as an **Erosion**
contact: it cuts into everything older and the cover is what lies above it.

* Surfaces only. No solids, no boolean operations.
* The base meets the topography exactly on the polygon edge (zero thickness there) and gets deeper inland.
* Outside the polygon the base lies above the topography, so in Leapfrog it cuts nothing there.

## 2. The depth model (one basin)

Let `d` be the signed distance to the polygon edge: positive inside, negative outside, zero on the edge.
Let `Z_top` be the DEM elevation.

* Inside (`d >= 0`):  `Z_base = Z_top - z_max * (1 - exp(-d / L))`. Depth is 0 on the edge and tends to `z_max`.
* Outside (`d < 0`): `Z_base = Z_top + h(|d|)` with `h = h_min * smoothstep(|d| / w) + h_slope * |d|`. The base rises
  above the ground, smoothly, and keeps rising with distance.

Parameters (metres; slopes in m/m) and how to choose them:

| Parameter | Meaning | Thin channels (about 50 m wide) | Large compact deposits |
|---|---|---|---|
| `z_max` | maximum depth | 15 | 80 |
| `L` | distance over which depth reaches about 63 % of `z_max` | 25 | 200 |
| `edge_spacing` | triangle edge on the polygon outline | 30 | 47 |
| `grid_cell` | cell of the distance grid (at most `edge_spacing`/3) | 10 | 20 |
| `h_min`, `ramp_width`, `h_slope` | lift above the ground outside the polygon | 3, 50, 0.30 | 5, 100, 0.25 |

Rule of thumb: `edge_spacing` should be at most half the width of the narrowest polygon you care about, and `z_max`
should be a plausible maximum thickness for that kind of cover. These are first guesses and are meant to be changed.

## 3. The mesh

Top and base share **one** graded constrained Delaunay triangulation of the whole bounding box. The constraints are
the polygon outline (the real basin edges) and the box edges; triangles are small near the edges and grow with
distance. Because the vertices are the same, the top and base surfaces correspond one to one.

## 4. A sequence of basins (the part that needs care)

Real cover comes in several units, each younger or older than the next. Treat them as an ordered list, **youngest
first**: basin A (youngest), basin B, basin C. Build them one after the other with these rules.

1. **Join touching polygons of the same basin** (dissolve them) so that there is no edge between them. An edge that
   is only a boundary between two parts of the same basin would otherwise pin the base to the ground there.
2. **Edges against a younger basin are not real edges of the older basin.** The older basin continues underneath the
   younger one. When building basin B, pass the polygons of basin A as `cover`. The builder then measures distance
   only to real edges and **fills the depth under the cover smoothly** by solving Laplace's equation from the known
   inside and outside values around it. Do **not** simply union the older basin with the younger one: that was
   tried and it failed to give a usable older basin.
3. **Never let two surfaces touch.** Two surfaces that cross exactly at shared vertices make Leapfrog's geometry
   engine fail ("CSG error", "too many bad_indices"). Pass the younger result as `younger` so that the older base is
   kept at least `gap` (1 m) below it wherever the older basin is deeper than zero, and build the older surface on a
   box that is 10 m **larger** and a different `edge_spacing` so that no vertex lands on another surface's vertex.
   Do **not** make the box smaller: the surface must cover the model boundary or Leapfrog reports an empty volume.
4. **Overlap in the source map.** If the map shows basin A polygons lying inside basin B polygons, basin B's polygon
   set is its own polygons only; the cover argument takes care of the rest.

Call (see `geomodeltools.basin`):

```python
from geomodeltools.basin import BasinParams, build_basin_surface, filled_outline

a = build_basin_surface(polys_A, crs, dem_path, bbox, BasinParams(z_max=15, L=25, edge_spacing=30, grid_cell=10),
                        out_dir="out", name="basin_A")
big = (bbox[0] - 10, bbox[1] + 10, bbox[2] - 10, bbox[3] + 10)
b = build_basin_surface(polys_B, crs, dem_path, big, BasinParams(z_max=80, L=200, edge_spacing=47, grid_cell=20),
                        out_dir="out", name="basin_B", cover=polys_A, younger=a)
outline_B = filled_outline(b)      # the basin outline as the surface sees it, including the part under basin A   # exact map edges + a smooth (marching-squares) edge only under the cover
```

Outputs per basin: `<name>_base.obj` and `<name>_top.obj` (real coordinates), `*_local.ply` (local frame),
`<name>_frame.json` (the local origin), `<name>_params_qa.json`, `<name>_transects.png`.

## 5. Checks before importing (all printed by the builder)

| Check | Expected |
|---|---|
| degenerate and flipped triangles | 0 and 0 |
| `ring_max_abs_dz` (base minus top on the real edges) | 0 (under 1e-6) |
| inside base below top, outside base above top | both true |
| minimum triangle angle | above about 3 deg; the 1st percentile above about 30 deg |
| `basin_volume_m3` | equals the sum of triangle area times mean depth; compare with `polygon_area * mean_depth` |
| shared vertices between two basins' meshes | 0 (count rows that appear in both `*_base.obj`) |

## 6. In Leapfrog (the Leapfrog side is detailed in agent-leapfrog's recipe)

1. Import each `*_base.obj` as a mesh (Meshes folder).
2. In the geological model's Surface Chronology, create **New Erosion > From Surface** for each basin, **youngest
   created last** (new surfaces appear at the top of the list; the top is the youngest).
3. First lithology = the basin (is **Younger**), second lithology = the background lithology (is Older).
4. Tick the surfaces. Process. Output volumes must be present and not "[Empty]".

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| older basin gives an empty volume, alone or with others | surface does not cover the model boundary (box made smaller), or the surface was set as Deposit | use a larger box; set the contact type to Erosion |
| "CSG error ... too many bad_indices" at points about 1e-7 m apart | two surfaces share vertices or one polygon outline is in two surfaces | nest the surfaces (section 4.3); do not put the same outline in two surfaces |
| a younger deposit has no volume | the older surface made of points on the ground sits at the topography and fills the space | use Erosion for cover; do not use Deposit "from points" for thin cover |
| the older basin closes on the younger polygon | younger edges were treated as basin edges | pass the younger polygons as `cover` |

## 8. Words used here

* **DEM**: digital elevation model, the topography as a grid of heights.
* **Signed distance**: distance to the nearest edge, positive inside the polygon, negative outside.
* **Erosion / Deposit contact (Leapfrog)**: an Erosion surface cuts older units; a Deposit surface is itself cut by
  older units.
* **CSG**: constructive solid geometry, the engine that Leapfrog uses to combine surfaces into volumes.
* **Younger side**: the side of a surface that holds the younger lithology; set from the surface normal.
