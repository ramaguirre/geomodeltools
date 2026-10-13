# Colours for geological units

`geomodeltools.palette`. Aim: units that are told apart at a glance in a 3D model, with colours a geologist expects.

## Convention
* **Layered units** (volcanic, pyroclastic, sedimentary): the colour of their age in the International Chronostratigraphic Chart
  (CGMW colour code), at stage level when the unit's age range fits one stage, otherwise at epoch level (`palette.ICS`).
  Several units in the same stage get a lightness ramp around the stage colour, youngest lightest (`palette.ramp_about`).
* **Intrusive units**: a red / pink family chosen by kind (plutonic, porphyry, hypabyssal), lighter when younger
  (`palette.family_ramp`), alternating a few degrees of hue between neighbours.
* **Unconsolidated cover**: a cream-to-ochre ramp from the Pleistocene colour, darker with age; made ground grey; water blue.
* The ICS values were taken from the Macrostrat copy of the ICS table (stage and epoch colours); check them against the current ICS
  chart or the CGMW RGB download before publishing a map.

## Check
`palette.delta_e2000` (CIEDE2000, tested against the published Sharma et al. pair) and `palette.min_pairwise_de` give the smallest
distance between any two units; the source map's own symbology had 0 (identical colours). A distance below about 5 is hard to
tell apart in 3D. `palette.spread_lightness` pushes a group apart in lightness only.

## Not decided by this module
Which unit gets which age colour is a geological decision made by the caller (see the project's table); intrusives are not coloured
by age hue on purpose, so that rock type reads first and age second.
