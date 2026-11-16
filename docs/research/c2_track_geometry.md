# C2 track geometry: sources for the 9 circuits missing from TUM

Date: 2026-11-15. Author: web-researcher. For: C2a geometry sourcing (`docs/specs/C2_style_critic.md` line 22; gate in `docs/specs/C2_review.md` R2).

Circuits: Baku, Imola, Las Vegas, Madrid (Madring), Miami, Monaco, Lusail, Jeddah, Marina Bay (current layout).
Target format: TUM `# x_m,y_m,w_tr_right_m,w_tr_left_m`, closed loop, driving direction, about 5 m spacing, local metric frame (example: `ocd_first_lap/vendor/racetrack-database/tracks/Monza.csv`, first row `-0.320123,1.087714,5.739,5.932`).

## 0. Up front: what this report does and does not deliver

- **No CSVs were produced and nothing was downloaded.** This agent has web search, web fetch (which returns a model summary of a page, not the raw bytes) and file read/write. It has no shell. So it cannot `curl` a file byte for byte, cannot checksum it, and cannot run a projection or spline. Copying coordinates through the summariser would not give 1 m precision and could not be verified. `data/tracks_extra/raw/` and `data/tracks_extra/*.csv` were therefore **not created**. Section 5 gives an exact, commit-pinned recipe so that rl-coder or sim-runner can do the download and conversion in one step.
- **No source has width or edges for any of the 9 circuits.** Every option below needs a width assumption or a measurement (Section 4).
- **No source has a computed racing line for any of the 9 in TUM `racelines` format.** The options are in Section 4.3.
- **Polyline lengths are not computed** (same reason: no shell). The "length check" column compares the length a source *states* with the official length. The real polyline length still has to be measured and must pass the C2 R2 gate (|L_game/L_geo − 1| ≤ 1 %).

## 1. Sources checked

### 1.1 TUM racetrack-database (baseline, for reference)
- Local copy `ocd_first_lap/vendor/racetrack-database/README.md`. Its 25 tracks include none of the 9.
- Method (README lines 35–37): "The original center lines were fetched as GPS points from the OpenStreetMap project … We applied a smoothing algorithm to the center lines. The track widths were extracted from satellite images using an image processing algorithm."
- Quality (README line 48): "The quality of the source data (GPS points and satellite images) varies greatly depending on the location."
- Licence: LGPL-3.0. I did not open the LICENSE file. This comes from the krrakenn/f1track DATA.md (1.6), which states "LGPL-3.0" for TUM.

### 1.2 f1tenth/f1tenth_racetracks: not useful
- https://github.com/f1tenth/f1tenth_racetracks
- 23 tracks, all derived from TUM, scaled 1:10 with "a fixed track width of 2.20 m". Licence GPL-3.0. None of the 9 is included.

### 1.3 bacinger/f1-circuits (GeoJSON): the only open source with all 9 centre lines
- Repo: https://github.com/bacinger/f1-circuits. Licence: MIT, "Copyright (c) 2019-2025 Tomislav Bacinger" (README).
- Master head at the time of checking: `8094e957b8d6aa8b9f4510c93968556cccd30e02` ("Formula 1™ circuits", 16 Oct 2026).
- Content per circuit: one GeoJSON `LineString` in WGS84 lon/lat. It is closed (first vertex equals last for all 9, checked). Properties are `id, Location, Name, opened, firstgp, length, altitude`.
- **No width, no edges, no racing line.**
- What the README does not say: how the lines were made, how accurate they are, which way they run, or what the `length` property means. It says only that the "Initial circuits data source" was a Google Maps page.
- My reading: the lines are hand-traced. The `length` property is the official figure typed in, not a measured length. Evidence: Lusail shows the pre-2023 figure 5380 and Madring the old 5474, while the geometry was edited later.
- **Accuracy evidence (from the commit history; the precision interpretation is mine):**
  - PR #6 (FinnLawrence, merged 24 Jul 2025), https://github.com/bacinger/f1-circuits/pull/6:
    - Baku: "Reversed order of coordinates to correctly represent counter-clockwise race direction". Direction was wrong before July 2025, so direction must be checked for every circuit.
    - Las Vegas: "Re-aligned circuit to follow the western (southbound) side of the boulevard, rather than the middle". This was a carriageway-sized error, roughly 10 m or more.
    - The PR did not touch the Koval Ln/Sphere/Sands Ave sections.
  - Baku commit `6d295da8b8a081c589c91d87bf2651fe7c24ce22` (16 Oct 2026) rewrote the coordinates. Example: [49.854798, 40.374319] → [49.854693, 40.374371], about 9–11 m.
  - So vertex accuracy is a few metres to about 10 m, not 1 m.
  - Vertex density is 84–160 points per lap, which is roughly 20–60 m apart on average. Curvature from these lines needs a smoothing spline (Section 5).
- Per-circuit data (fetched from `raw.githubusercontent.com/bacinger/f1-circuits/master/circuits/<id>.geojson`; dates from each file's GitHub commit history):

| Circuit | file | vertices | stated `length` | last geometry commit |
|---|---|---|---|---|
| Baku | az-2016.geojson | 86 | 6003 | 16 Oct 2026 (6d295da, 1d059c1). Direction reversed Jul 2025 |
| Imola | it-1953.geojson | 84 | 4909 | 8 Nov 2020 (13c8517), the only commit shown |
| Las Vegas | us-2023.geojson | 100 | 6201 | 5 Aug 2025 (928c645). Earlier Jul 2025 corrections (884c81f, 4501c19) |
| Madrid | es-2026.geojson | 116 | 5474 | 11 Jun 2025 (93d12d1), the only commit |
| Miami | us-2022.geojson | 103 | 5412 | 29 Jul 2025 (c7df018) |
| Monaco | mc-1929.geojson | 160 | 3337 | 16 Jan 2024 (b8c7e17) |
| Lusail | qa-2004.geojson | 107 | 5380 | 25 Jul 2025 (54d2ad3) |
| Jeddah | sa-2021.geojson | 152 | 6175 | 16 Mar 2023 (4f377b0). 15 Mar 2023: "2023 track update based on the latest satellite images" |
| Singapore | sg-2008.geojson | 116 | 4928 | 8 Sep 2023 (dcba544). 11 Nov 2022: "2023 track update" |

### 1.4 OpenStreetMap: raceway ways and circuit relations
- Licence: ODbL (OpenStreetMap contributors).
- Wiki https://wiki.openstreetmap.org/wiki/Tag:highway%3Draceway: the page does not mention a width tag. `type=circuit` is "A proposed relation type than can be used for mapping on-street circuits". `area:highway=raceway` is a proposed tag ("This is controversial").
- **Width is rarely tagged.** taginfo (https://taginfo.openstreetmap.org/api/4/tag/combinations?key=highway&value=raceway&filter=ways) shows `width` on 1,363 ways, a to_fraction of 0.0296 (about 3 %) of `highway=raceway` ways.
- Relations and ways found (OSM API v0.6 and Nominatim, checked 15 Nov 2026):
  - **Monaco**: relation 148194, `type=circuit`, `length=3337`, version 61, 2026-08-21. Members: 42 ways (including 1 `pit_lane` role) and 1 `start-finish` node. Members are public-road ways (secondary/residential/service). Of the 22 member ways I could read, **0 have `width`** and 14 have `lanes=2`.
  - **Singapore**: relation 421263, `type=circuit`, version 107, 2026-05-14, 100 ways + 2 nodes. Member way 686335807 is `highway=raceway`, `oneway=yes`, with no width.
  - **Las Vegas**: relation 16696508, `type=circuit`, version 24, 2026-08-04, 101 members. Way 1456078477 is `highway=raceway`, `oneway=yes`, with no width.
  - **Madrid**: relation 18813472, `route=raceway`, `type=route`, `name=MadRing`, version 8, **2026-09-16** (after the September 2026 race), 23 members. Way 1366550177 has 98 nodes and no width.
  - **Jeddah**: raceway ways 1257884699 and 1007989410, with no circuit relation on either. **Way 1257884699 carries `fixme=Formula E version`** (version 6, 2026-03-07). A Formula E layout overlaps here, so member selection must be checked.
  - **Imola, Lusail, Miami**: Nominatim returned only the facility polygons (Imola relation 13706843, Lusail way 152483594, Miami way 1017340360, all `leisure=sports_centre`). Raceway ways were not checked because the Overpass API timed out or returned 502 on four endpoints.
  - **Baku**: Nominatim returned only a carport building. Raceway ways and circuit relation were not checked.
- My reading:
  - OSM gives a denser centre line than bacinger where dedicated `highway=raceway` ways exist (Singapore, Las Vegas, Madrid, Jeddah).
  - On public-road relations (Monaco) it gives *road* centre lines, which can differ from the track centre where the circuit uses only part of a carriageway.
  - Assembling relation members into an ordered loop (dropping pit lanes) is extra work.
  - No widths.

### 1.5 FastF1 / OpenF1 car positions: centre-line shape yes, edges no
- FastF1 (MIT, https://github.com/theOehrly/Fast-F1; docs https://docs.fastf1.dev/core.html):
  - Position channels: "X (float): X position [1/10 m]", same for Y and Z.
  - The "How to perform calculations accurately" page says the F1 API data has "a very limited sample rate of about 4-5 Hz" and is "somewhat jittery". It warns "Do not resample the input data" and "Do not perform calculations based on values that have been calculated by means of integration".
  - `Session.get_circuit_info()` gives corner numbers and positions in the same frame, from MultiViewer. The docs say: "This data has been manually created and is not highly accurate but sufficient for visualization."
- OpenF1 (https://openf1.org/docs/), location endpoint:
  - "The approximate location of the cars on the circuit, at a sample rate of about 3.7 Hz … Useful for gauging their progress along the track, but lacks details about lateral placement — i.e. whether the car is on the left or right side of the track."
  - Origin "arbitrary". Site content is "Licensed under CC BY-NC-SA 4.0". Historical data since 2023 is free.
- FastF1 developer (Oehrly, f1technical.net forum, 25 Jul 2020; old, but no later contradiction found): "You cannot get the on-track position or choice of line in a corner from the data." https://www.f1technical.net/forum/viewtopic.php?p=912666
- **Conclusion (my reading):**
  - The user's idea of taking edges from the envelope of many cars is **not supported**. Both primary sources say lateral placement is not resolved.
  - Positions are fine for the **shape and length** of the driven loop, the **driving direction**, the start/finish location and corner positions.
  - That makes them the best available check of bacinger/OSM geometry: ICP alignment plus a length check against FastF1's 1/10 m scale.
  - Data availability (my reading of the calendars; not checked session by session):
    - 2023–2026 races: all 9 circuits except Madrid (2026 only) and Imola (last race 2025).
    - Las Vegas: 2023–2025.
    - Singapore 2025+ layout: 2025–2026.
- Underlying data is from F1's live-timing feed: unofficial, research and education use.

### 1.6 krrakenn/f1track (GitHub): useful as method, not as data
- https://github.com/krrakenn/f1track: "Physics-based optimal racing lines for every 2026 F1 circuit". The page shows 1 commit and © 2026.
- Method (docs/DATA.md):
  - TUM where the layout matches. Otherwise OSM `highway=raceway` ways and `type=circuit` relations "in a ±0.03° box".
  - Widths from `nominal_width_m` in `config/circuits.yaml`, "widened to observed car positions".
  - README: "Madring geometry is approximate because OSM data is incomplete."
- Nominal widths in `config/circuits.yaml`: Baku 10.0, Las Vegas 12.0, Madring 12.0, Miami 12.0, Monaco 9.0, Lusail 12.0, Jeddah 11.0, Singapore 10.0 m. Imola is absent.
- **No licence file was found** (all rights reserved by default), so do not copy its data. Its widths are author assumptions with no stated source, not measurements.

### 1.7 F1 game UDP telemetry: the only route to 1 m edges that matches the game
- F1 23 spec as reproduced in https://github.com/MacManley/f1-23-udp (MIT):
  - `float m_worldPositionX; // World space X position - metres`. Y and Z have no unit comment.
  - `uint16 m_trackLength; // Track length in metres`
  - `float m_lapDistance; // Distance vehicle is around current lap in metres`
  - Motion packet "Frequency: Rate as specified in menus"
- The F1 25 / F1 25 2026 Season Pack spec is on the EA forums (https://forums.ea.com/blog/f1-games-game-info-hub-en/f1%c2%ae-25-udp-specification/12187347). **It returned 403, not opened.** A community post (search snippet only) lists F1 24→25 changes that do not involve the world-position fields.
- Which axis is vertical is **not confirmed**. Forum posts disagree: some plot X/Z, others X/Y. Determine it empirically, as the axis with the smallest range.
- My reading:
  - Recording one slow lap along each edge in Time Trial gives game-consistent edges at metre level, plus `m_trackLength` for the R2 gate.
  - Recording AI or own fast laps at 60 Hz gives a real driven line.
  - This needs Ibrahim's game time (about 3 laps × 9 tracks). The C2 telemetry laps have no positions (C2 spec line 54), so this would be new recording.

### 1.8 Not pursued
- Sim-racing mod tracks (Assetto Corsa AI splines, rFactor AIW) do store per-point left/right widths. Mods have no open licence as a rule, and extracting them means parsing binary game files. I did not check any.
- Google/Esri imagery measurement is a manual option (Section 4). Imagery terms of use apply.
- FIA licensed-circuit lists (https://www.fia.com/sites/default/files/circuits_fia20260331_0.pdf) and FIA Appendix O 2026 PDF: the fetch returned compressed PDFs and the local PDF reader is unavailable, so **not opened**. A search snippet suggests the circuit list has lap distance and grade but no width.

## 2. Official lengths and layout status (for the length check)

| Circuit | official current length | source | layout notes |
|---|---|---|---|
| Baku | 6.003 km | F1.com circuit guide, 23 Sep 2026: "Circuit length: 6.003km" | narrowest "7.6m wide" at T8/9 (same article) |
| Imola | 4.909 km | Wikipedia infobox: Grand Prix Circuit 2008–present | not on the 2026 calendar |
| Las Vegas | 6.201 km | Wikipedia infobox, 2023–present | bacinger Sphere section not re-traced in PR #6 |
| Madrid | **5.414 km**, 22 turns | F1.com circuit guide, 10 Sep 2026 | bacinger states 5474 (older IFEMA figure). Wikipedia: "22 corners (originally announced as 20)". **Layout match of bacinger es-2026 (Jun 2025) unverified** |
| Miami | 5.412 km | Wikipedia infobox, 2022–present | |
| Monaco | 3.337 km | Wikipedia infobox, 2015–present; OSM relation `length=3337` | |
| Lusail | **5.419 km** | Wikipedia: 5.419 km 2023–present, 5.380 km 2004–2022. Autosport: 2023 resurfacing with no layout change | bacinger states 5380 (stale) |
| Jeddah | 6.174 km | Wikipedia infobox, 2021–present | bacinger 6175. Wall/sightline changes 2022–23; bacinger updated Mar 2023 |
| Singapore | **4.927 km** (2025–present); 4.940 km (2023–24) | Wikipedia infobox | 2023 change replaced T16–19 with a straight. Wikipedia gives no reason for the 2025 13 m change. bacinger states 4928, geometry updated Nov 2022 / Sep 2023 |

## 3. Per-circuit assessment

Shared by all 9:
- **Centre line:** bacinger (MIT), and for some circuits OSM (ODbL).
- **Width/edges:** none.
- **Racing line:** none.
- **Status:** none ready; all need conversion, a width decision and a measured-length check.
- **Recommended check for each:** align the converted line to a FastF1 fastest-lap trace (ICP, rotation and translation only). This confirms layout, direction and length, then the C2 R2 gate applies.

Circuit-specific notes:
- **Baku:** the most recently edited bacinger file (Oct 2026). Narrow castle section, 7.6 m minimum (F1.com), so a constant 12 m width would be wrong there.
- **Imola:** bacinger file untouched since 2020. My reading is that it is low risk because the layout has not changed since 2008 per Wikipedia. Vertex accuracy is unknown.
- **Las Vegas:** bacinger was corrected in 2025. The OSM circuit relation (2026) is an independent alternative.
- **Madrid:** highest risk.
  - bacinger predates the final layout: its stated length is 5474 against an official 5414, and the corner count changed 20 → 22.
  - Prefer OSM relation 18813472, updated 16 Sep 2026, or the FastF1 2026 Spanish GP mean line. Validate against F1.com's 22-turn layout.
  - The banked La Monumental corner (≤ 24 % gradient) is irrelevant in 2D.
- **Miami:** low risk. Note the extra 2024 club layouts in OSM; do not pick the wrong member ways.
- **Monaco:**
  - bacinger has the densest file (160 vertices). The OSM relation is road centre lines with no width.
  - No reliable width figure was found. One fan-site figure ("about 7 meters wide … Loews Hairpin") is uncorroborated and **not used**.
- **Lusail:** check the measured polyline length against 5.419 km, not the stated 5380.
- **Jeddah:** watch the Formula E overlap in OSM (`fixme=Formula E version`).
- **Singapore:** bacinger has the 2023+ straight. Whether it reflects the 2025 4.927 km change is unknown, but that is only a 13 m difference.

## 4. Width, edges and racing line: options (none chosen; each is an assumption to be approved)

### 4.1 Width options
| id | option | basis | precision | cost |
|---|---|---|---|---|
| W1 | Constant 12 m, `w_tr_right_m = w_tr_left_m = 6.0` | FIA Appendix O 2026 (published 23 Dec 2025), seen **only as a search snippet**: "The track width foreseen should be at least 12 m" for new circuits. Not opened; street circuits have exceptions (secondary sources). | assumption; wrong on narrow street sections (Baku 7.6 m min) | zero |
| W2 | Constant = median total width over the 25 TUM tracks | computed from local data (e.g. Monza row 1: 5.739 + 5.932 = 11.67 m). sim-runner computes the median. | assumption, but grounded in measured F1 tracks | minutes |
| W3 | Per-circuit constants, street circuits narrower | only Baku has a cited figure (7.6 m minimum, F1.com). Others would be guesses. krrakenn's numbers are unlicensed author guesses. | assumption | zero |
| W4 | Measure on satellite imagery at stations (every corner + straights, interpolate) | TUM's own approach (automated there) | ~1 m | hours per track; imagery terms |
| W5 | Record edges in F1 25 via UDP (slow lap hugging each edge) | game world position "metres" (F1 23 spec); also gives `m_trackLength` | ~1 m, and matches the game the setups came from | ~3 laps per track of Ibrahim's time |
| ✗ | FastF1 / OpenF1 envelope | primary sources say lateral placement is not resolved (1.5) | not usable | |

### 4.2 Effect of width on the C2 features (my reading)
- Corners from curvature (`Track.find_corners()`) do not depend on width.
- The placeholder line's lateral offsets and distances to edges, and a minimum-curvature line, do depend on width. Under W1–W3 these numbers are artefacts of the assumption, which fits their "placeholder" flag (C2 spec line 30).

### 4.3 Racing line options (TUM `racelines` format `# x_m,y_m`)
- R-a: minimum-curvature line from TUMFTM global_racetrajectory_optimization (https://github.com/TUMFTM/global_racetrajectory_optimization, the same algorithm TUM used for its racelines; LGPL-3.0 per 1.1) on the converted centre line plus the chosen width.
- R-b: FastF1 driven line, the median of the fastest laps. It is real but laterally unreliable per 1.5, so use it for shape and length, not lateral placement.
- R-c: F1 25 UDP recorded line (W5 session).

## 5. Conversion recipe (for rl-coder / sim-runner; not executed)

1. **Download** (curl, no code from the download):
   - URL: `https://raw.githubusercontent.com/bacinger/f1-circuits/8094e957b8d6aa8b9f4510c93968556cccd30e02/circuits/<id>.geojson`
   - Save into the new empty directory `data/tracks_extra/raw/bacinger_f1-circuits_8094e95/`.
   - Record the sha256 of each file.
   - OSM alternative: an Overpass query `relation(<id>);way(r);out geom;` (Monaco 148194, Singapore 421263, Las Vegas 16696508, Madrid 18813472) saved into `data/tracks_extra/raw/osm_<date>/`. Attribution "© OpenStreetMap contributors, ODbL".
2. **Parse** with `python -I` (json only). Drop the duplicate closing vertex.
3. **Project** lon/lat to a local metric frame:
   - Azimuthal equidistant centred on the polyline centroid (pyproj `+proj=aeqd +lat_0=… +lon_0=… +datum=WGS84`), or UTM.
   - An equirectangular cos(lat₀) approximation is also acceptable. My estimate is under 1 m error within 3 km of the centre.
   - Record the CRS string.
4. **Direction:** make the vertex order follow the race direction. Check it against the FastF1 trace (lap-time order of positions) after ICP alignment that allows rotation and translation only. Do not take direction from memory; bacinger had Baku reversed until 2025.
5. **Start:** rotate the loop so that index 0 is the start/finish line. Use the OSM `start-finish` node where present, or the FastF1 lap-start position.
6. **Smooth and resample:**
   - Fit a periodic smoothing spline (e.g. `scipy.interpolate.splprep(per=True)`).
   - Choose the smoothing so the RMS deviation from the vertices is about 2 m, consistent with the bacinger edit sizes in 1.3. Record `s` and the RMS/max deviation.
   - Resample at 5 m arc length.
7. **Width:** write `w_tr_right_m, w_tr_left_m` from the approved option in 4.1. Record the option id in a sidecar provenance JSON (source URL, commit, sha256, CRS, smoothing, width option, length).
8. **Checks:**
   - Measured length vs official (Section 2) and vs `m_trackLength`/game lap length (C2 R2: ≤ 1 %).
   - Curvature corner count vs the F1.com/Wikipedia turn count.
   - FastF1 ICP residual (report the median and 95th-percentile distance).
9. Write `data/tracks_extra/<CircuitName>.csv` with the header `# x_m,y_m,w_tr_right_m,w_tr_left_m`.

## 6. Summary table

| circuit | source | centre line | width/edges | racing line | licence | length check (stated vs official) | ready? |
|---|---|---|---|---|---|---|---|
| Baku | bacinger az-2016 @8094e95; OSM not found | yes, 86 vertices, ~m–10 m accuracy, edited Oct 2026 | none (7.6 m min, F1.com) | none | MIT | 6003 vs 6.003 km ✓; polyline not measured | no: convert + width + check |
| Imola | bacinger it-1953 | yes, 84 vertices, last edit 2020 | none | none | MIT | 4909 vs 4.909 ✓; not measured | no |
| Las Vegas | bacinger us-2023; OSM rel 16696508 | yes, 100 vertices (2025 fixes); OSM alternative | none | none | MIT / ODbL | 6201 vs 6.201 ✓; not measured | no |
| Madrid | bacinger es-2026 (Jun 2025); OSM rel 18813472 (Sep 2026) | bacinger likely outdated; OSM preferred | none | none | MIT / ODbL | 5474 vs **5.414** ✗ (stated); not measured | no: layout must be verified |
| Miami | bacinger us-2022 | yes, 103 vertices | none | none | MIT | 5412 vs 5.412 ✓; not measured | no |
| Monaco | bacinger mc-1929; OSM rel 148194 (road ways) | yes, 160 vertices | none (no reliable figure) | none | MIT / ODbL | 3337 vs 3.337 ✓; not measured | no |
| Lusail | bacinger qa-2004 | yes, 107 vertices | none | none | MIT | 5380 vs **5.419** ✗ (stated value stale); not measured | no |
| Jeddah | bacinger sa-2021; OSM raceway ways (FE overlap) | yes, 152 vertices (2023 update) | none | none | MIT / ODbL | 6175 vs 6.174 ✓; not measured | no |
| Singapore | bacinger sg-2008; OSM rel 421263 | yes, 116 vertices, 2023+ straight | none | none | MIT / ODbL | 4928 vs 4.927 (2025+) ✓; not measured | no |

## 7. Implications for us (interpretation)
1. Centre lines for all 9 are available today under MIT (bacinger). Conversion is a short scripted job (Section 5) for rl-coder/sim-runner, followed by the R2 gate.
2. Width is the real gap. W2 (TUM median) is the cheapest defensible placeholder. W5 (record in F1 25) is the only route to roughly 1 m edges that also match the game's geometry and length. That choice is Ibrahim's.
3. Do not build edges from FastF1/OpenF1 envelopes. Use FastF1 only to check layout, direction and length.
4. Madrid and Lusail need the most care: a stale or possibly outdated layout or length. Baku and Monaco make any constant-width assumption visibly wrong in narrow sections.
5. The same recipe covers the 5 TUM tracks that failed the R2 gate (Abu Dhabi, Mexico, Netherlands, Melbourne, Catalunya), since bacinger has all of them. Not checked here.

## Sources
- TUM racetrack-database README (local vendor copy): /Users/ibrahimkhaled/Desktop/RESEARCH/ocd_first_lap/vendor/racetrack-database/README.md
- bacinger/f1-circuits: https://github.com/bacinger/f1-circuits ; PR #6 https://github.com/bacinger/f1-circuits/pull/6 ; commit 6d295da https://github.com/bacinger/f1-circuits/commit/6d295da ; commit 8094e95 https://github.com/bacinger/f1-circuits/commit/8094e95
- f1tenth_racetracks: https://github.com/f1tenth/f1tenth_racetracks
- OSM wiki highway=raceway: https://wiki.openstreetmap.org/wiki/Tag:highway%3Draceway ; taginfo combinations: https://taginfo.openstreetmap.org/api/4/tag/combinations?key=highway&value=raceway&filter=ways ; OSM API relations 148194, 421263, 16696508, 18813472; ways 1366550177, 1456078477, 1257884699, 686335807 (https://api.openstreetmap.org/api/0.6/...)
- FastF1 docs: https://docs.fastf1.dev/core.html ; https://docs.fastf1.dev/data_reference/howto_accurate_calculations.html ; repo https://github.com/theOehrly/Fast-F1
- OpenF1 docs: https://openf1.org/docs/ ; licence footer https://openf1.org/
- Oehrly forum posts: https://www.f1technical.net/forum/viewtopic.php?p=912666
- krrakenn/f1track: https://github.com/krrakenn/f1track ; docs/DATA.md ; config/circuits.yaml
- F1 23 UDP spec reproduction: https://github.com/MacManley/f1-23-udp ; EA F1 25 spec (not opened, 403): https://forums.ea.com/blog/f1-games-game-info-hub-en/f1%c2%ae-25-udp-specification/12187347
- F1.com circuit guides: Baku https://www.formula1.com/en/latest/article/circuit-guide-everything-you-need-to-know-about-the-baku-city-circuit.320UGBNQu2ALdgAtax1gRn ; Madring https://www.formula1.com/en/latest/article/circuit-guide-everything-you-need-to-know-about-the-madring.NF7Mh3iag3w9GUPlihwJA.NF7Mh3iag3w9GUPlihwJA
- Wikipedia infoboxes: Marina Bay https://en.wikipedia.org/wiki/Marina_Bay_Street_Circuit ; Jeddah https://en.wikipedia.org/wiki/Jeddah_Corniche_Circuit ; Las Vegas https://en.wikipedia.org/wiki/Las_Vegas_Strip_Circuit ; Miami https://en.wikipedia.org/wiki/Miami_International_Autodrome ; Madring https://en.wikipedia.org/wiki/Madring ; Imola https://en.wikipedia.org/wiki/Imola_Circuit ; Monaco https://en.wikipedia.org/wiki/Circuit_de_Monaco ; Lusail https://en.wikipedia.org/wiki/Lusail_International_Circuit
- Lusail 2023 works (no layout change): https://www.autosport.com/f1/news/losail-upgraded-as-qatar-promises-f1-race-will-be-annual-world-cup/10474093/
- FIA Appendix O 2026 (not opened, snippet only): https://www.fia.com/system/files/documents/appendix_o_2026_published_23122025.pdf
- AE6310 course notes (10–15 m width suggestion, uncited, not used): https://smdogroup.github.io/ae6310/jupyter/2026_project/race_car_problem.html
