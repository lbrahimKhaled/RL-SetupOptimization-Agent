# C2 follow-up, 15 Nov 2026: game setup units, layouts, steering, toe

Author: web-researcher. Companion to `docs/research/c2_track_geometry.md`. Item 2 (layouts) is placed here too, because this agent can only overwrite files, not append.

Limits that apply to everything below:
- The official EA F1 25 / 2026 Season Pack UDP specification (https://forums.ea.com/blog/f1-games-game-info-hub-en/f1%c2%ae-25-udp-specification/12187347) returned HTTP 403. **Not opened.** Struct quotes come from the F1 23 specification as reproduced in https://github.com/MacManley/f1-23-udp (MIT). Whether F1 25/26 changed these comments is unverified.
- No official EA/Codemasters statement on physical units of setup sliders was found.

## 1. Setup menu values: physical units

**Ride height: mm is not confirmed anywhere I could find.**
- UDP `CarSetupData` (F1 23 spec): `uint8 m_frontSuspensionHeight; // Front ride height` and `uint8 m_rearSuspensionHeight; // Rear ride height`. No unit in the comment. Other fields in the same struct do carry units ("(percentage)", "(PSI)"), so the omission looks deliberate (my reading).
- Community guides give bare numbers with no unit, for example traxion.gg (6 Jun 2025): "In F1 25, a front ride height of 20 works well with a rear ride height of 45."
- The ranges 15–35 front / 40–60 rear were **not confirmed** by any source found. One F1 24 guide (search snippet, not opened) lists front "Min 10 – 40 Max" and rear "Min 40 – 100 Max", so ranges differ between game years; take them from the data, not from memory.
- One click: not documented. The field is a `uint8`, so one click = 1 unit of an unlabelled scale (my reading).

**Suspension 1–41, anti-roll bar 1–21, wings 0–50: no physical values published.**
- UDP comments: `uint8 m_frontWing; // Front wing aero`, `uint8 m_frontSuspension; // Front suspension`, `uint8 m_frontAntiRollBar; // Front anti-roll bar` (the rear ARB line carries the same "Front anti-roll bar" comment, a typo in the source). No units, no N/mm, no degrees.
- traxion.gg only gives direction: suspension "stiff settings (higher numbers)", anti-roll bar "Stiffer settings (higher numbers)", wings "Higher values equal more downforce."
- The ranges 1–41, 1–21 and 0–50 themselves were not confirmed by a source; example values seen (41, 21, wing 36) are consistent with them.

**Fields that do have units in the spec:** differential on/off throttle "(percentage)", brake pressure and bias "(percentage)", tyre pressures "(PSI)", camber and toe as float "angle (suspension geometry)" shown in degrees in the menu (traxion.gg lists "Front Camber: -3.50°").

## 2. Layout versions: game vs TUM files

**Game (official Codemasters coverage, real layout lengths from Wikipedia infoboxes):**

| Track | Game layout | Evidence | Real length of that layout |
|---|---|---|---|
| Albert Park | 2022 layout (2021–present on Wikipedia). Laser-scanned for F1 25. | F1 22 press coverage (May 2022): "new layout for turns 1, 3, 6, 8/9, 11, 12 and 13". traxion.gg 17 Apr 2025: "Albert Park, Melbourne (Australia) – Laser-scanned". | 5.278 km, 14 turns |
| Catalunya | No final chicane, new Turn 10 | F1 22 added the 2021 Turn 10; traxion.gg: "Up-to-date Spanish and Singapore Grand Prix layouts coming to F1 23" (no-chicane layout at launch). | 4.657 km, 14 turns |
| Yas Marina | Post-2021 layout | F1 22 press coverage: hairpin moved, chicane before it removed, hotel section reprofiled. | 5.281 km, 16 turns |
| Zandvoort | Not verified by a source. The circuit entered the series with F1 2020, so the 2020 rebuild is the likely layout (my inference). | none | 4.259 km, 14 turns |
| Mexico City | No layout note found in any F1 22–25 track list. | none | 4.304 km, 17 turns |

I found nothing that says F1 25 or F1 26 changed any of these five beyond the Melbourne rescan.

**TUM files:**
- The `tracks` folder was committed 6–7 May 2020 ("Added track data and readme", `1764427`; "reworked lines such that origin is on the S/F line", `2b44832`); only IMS was added later (4 Jun 2020). The repo has no issues. https://github.com/TUMFTM/racetrack-database/commits/master/tracks
- The README does not state a layout year for any file. The centre lines are OpenStreetMap as of May 2020 or earlier.
- So, as inference from the dates (not stated by TUM):
  - `Melbourne.csv`: 1996–2020 layout, 5.303 km, 16 turns. Does not match the game.
  - `Catalunya.csv`: 2007–2020 layout with chicane and old Turn 10, 4.655 km. Does not match the game. The length differs by only 2 m, so the length test alone will not catch it; the curvature correlation will.
  - `YasMarina.csv`: 2009–2021 layout, 5.554 km, 21 turns. Does not match the game.
  - `Zandvoort.csv`: ambiguous. Either the 1999–2019 layout (4.307 km) or the 2020 rebuild (4.259 km), depending on when OSM was updated. Measuring the file length decides it.
  - `MexicoCity.csv`: the 2015–present layout (4.304 km) should be what OSM had. A layout mismatch is not the expected reason for its gate failure.
- I did not measure the five file lengths (no shell). The coder should print them next to the figures above.

**Mexico: game distance about 4.52 km vs 4.304 km official. No explanation found.**
- No source mentions a longer in-game Mexico lap. Setup sites list 4.304 km for the game track.
- No real layout is 4.52 km. Nearest historical ones: 4.484 km (2002–2005), 4.463 km (2006–2007), 4.421 km (1986–2014). None is in the game as far as I can tell.
- Checks that would settle it (suggestions, not findings): read `m_trackLength` from the UDP session packet for Mexico; compare the last `m_lapDistance` of a lap with it; check whether the 4.52 km figure appears for all Mexico laps or comes from the distance channel of the lap source (a per-source scale error would show as the same ratio, about 1.05, on every lap).

## 3. f1laps.com steering channel

- **Not verified that it is raw `m_steer`.** F1Laps documents the trace only as: "Shows steering wheel angle", centre line "Straight ahead", peaks "Maximum steering lock" (https://www.f1laps.com/faqs/read-telemetry-data/). No unit, no range, no statement on raw vs assisted.
- F1Laps does say its data comes from the game's UDP stream (the telemetry app "receives the UDP telemetry stream"; position comes from "UDP Motion"). The only steering field in the UDP car-telemetry packet is `float m_steer; // Steering (-1.0 (full lock left) to 1.0 (full lock right))` (F1 23 spec). So the trace is very likely `m_steer`, possibly rescaled (my inference). Check the range of the stored values: if it is within ±1, it is unscaled.
- **Wheel angle at 1.0: not published.** The spec defines 1.0 as "full lock", not degrees. Community wheel guides say F1 cars use "around 360° rotation in total (180° each direction)" and recommend 360° for the in-game maximum wheel rotation, some 380–390° (simracingsetup.com, traxion.gg; third-party, not EA). Whether full lock in the physics equals ±180° at the wheel, and the road-wheel angle it maps to, could not be verified.
- **Device input or post-assist: not documented.** The spec comment does not say. With a pad, the game filters stick input before it reaches the car, and I found no statement on which side of that filter `m_steer` sits.

## 4. Toe convention

- The F1 25 setup menu labels the two values **"Front Toe-Out"** and **"Rear Toe-In"**. traxion.gg (6 Jun 2025) lists "Front Toe-Out: 0.00°" and "Rear Toe-In: 0.10°".
- So both are unsigned magnitudes in degrees: a larger front value means more toe-out, a larger rear value means more toe-in. In a toe-in-positive convention the front value takes a minus sign.
- The UDP fields are `float m_frontToe; // Front toe angle (suspension geometry)` and `float m_rearToe; // Rear toe angle (suspension geometry)`: no sign convention stated.
- The ranges front 0–0.2 and rear 0.1–0.25 were not confirmed by a source; community values (0.00–0.08 front, 0.10–0.23 rear) fall inside them. One F1Laps page shows a front toe of 0.5°, which is outside (search snippet, not opened; possibly a different game year).
- Not verified: whether the value is per wheel or total, and whether F1 26 keeps the same labels.

## Implications for us (interpretation)
1. Treat ride height, suspension, anti-roll bar and wing values as ordinal clicks on an unlabelled scale. Any mm or N/mm mapping is an assumption and needs Ibrahim's approval; nothing published supports one.
2. Toe can be given a sign from the menu labels: front = toe-out, rear = toe-in.
3. For Melbourne, Catalunya and Yas Marina the TUM files predate the layouts the game uses, which is consistent with the R2 gate failures. bacinger/f1-circuits is the replacement candidate (same recipe as the 9 circuits). Zandvoort needs a length measurement first. Mexico's failure is unexplained and looks like a distance-scale problem on the game side rather than geometry.
4. Use steering only as a normalised signal (sign and shape, as in the corr(steer, κ) gate). Do not convert it to degrees without a measured reference.

## Sources
- F1 23 UDP spec reproduction: https://github.com/MacManley/f1-23-udp
- EA F1 25 UDP spec (403, not opened): https://forums.ea.com/blog/f1-games-game-info-hub-en/f1%c2%ae-25-udp-specification/12187347
- traxion.gg F1 25 setup guide (6 Jun 2025): https://traxion.gg/f1-25-setup-guide/
- traxion.gg F1 25 track list (17 Apr 2025): https://traxion.gg/f1-25-track-list/
- F1 22 circuit updates: https://www.thesixthaxis.com/2022/05/19/f1-22-track-updates-for-spain-australia-abu-dhabi-illustrated/ ; https://traxion.gg/watch-the-major-track-changes-in-the-f1-22-game/ (search summaries, pages not opened)
- F1 23 Barcelona layout: https://traxion.gg/up-to-date-spanish-and-singapore-grand-prix-layouts-coming-to-f1-23/ (search summary, page not opened)
- TUM commits: https://github.com/TUMFTM/racetrack-database/commits/master/tracks
- Wikipedia infoboxes: https://en.wikipedia.org/wiki/Albert_Park_Circuit ; https://en.wikipedia.org/wiki/Circuit_de_Barcelona-Catalunya ; https://en.wikipedia.org/wiki/Yas_Marina_Circuit ; https://en.wikipedia.org/wiki/Circuit_Zandvoort ; https://en.wikipedia.org/wiki/Aut%C3%B3dromo_Hermanos_Rodr%C3%ADguez
- F1Laps: https://www.f1laps.com/faqs/read-telemetry-data/ ; https://www.f1laps.com/blog/f1-racing-line-telemetry/ ; https://www.f1laps.com/faqs/f1-25-udp-telemetry-settings/
- Wheel rotation guides (third-party): https://simracingsetup.com/ea-sports-f1/f1-25-wheel-settings/ ; https://traxion.gg/best-moza-wheelbase-settings-for-f1-25/ (search summaries, pages not opened)
