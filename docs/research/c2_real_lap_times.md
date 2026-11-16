# Real-world F1 reference lap times (2025, 2026) for checking game lap times

Compiled 16 Nov 2026. Purpose: give real reference laps (pole, fastest race lap) so we can check F1 25 / F1 25 2026 Season Pack ("F1 26") lap times.

## Sources and how the values were checked

- **Primary database: Jolpica-F1 API** (the community successor to Ergast, which serves the same data schema). It was queried on 2026-11-16:
  - Poles: `https://api.jolpi.ca/ergast/f1/{2025,2026}/qualifying/1.json` (pole = the Q3 time of the P1 qualifier).
  - Fastest race laps: `https://api.jolpi.ca/ergast/f1/{2025,2026}/fastest/1/results.json` (FastestLap rank 1).
  - Barcelona history: `https://api.jolpi.ca/ergast/f1/circuits/catalunya/qualifying/1.json` and `.../circuits/catalunya/fastest/1/results.json`.
- **Spot checks against independent sources.** Every spot-checked value matched Jolpica:
  - 2026 Barcelona pole 1:14.679: FIA, https://www.fia.com/news/f1-russell-storms-barcelona-pole-ahead-hamilton-and-antonelli-leclerc-crashes (dated 13.06.26; top 3 are 1:14.679 / 1:14.743 / 1:14.998).
  - 2026 Barcelona fastest lap (Hamilton): https://www.lightsoutblog.com/2026-barcelona-grand-prix-statistics/ says "Hamilton set the fastest lap of the race." It does not give the time. The 1:20.122 (lap 44) value is from Jolpica only.
  - 2025 Barcelona pole 1:11.546 and fastest lap 1:15.743 (Piastri, lap 61): FIA qualifying news (https://api.fia.com/news/f1-piastri-grabs-pole-spain-mclaren-lock-out-front-row-ahead-verstappen) and the FIA race-fastest-laps PDF (https://api.fia.com/sites/default/files/2025_09_esp_f1_r0_timing_racefastestlaps_v01.pdf). The PDF is image/binary and I could not extract its text myself. The search engine's extract quotes "1:15.743 on lap 61, 221.343 km/h".
  - 2025 Miami pole 1:26.204 (Verstappen): Jolpica has **empty** Q1/Q2/Q3 strings for this race. I took the time from Crash.net, https://www.crash.net/f1/results/1069184/1/2025-f1-miami-grand-prix-qualifying-results (Q3 "1m26.204s").
  - 2026 Monaco pole 1:12.051, 2026 Monza pole 1:21.786 (Gasly), 2026 Madring pole 1:31.824, 2026 Interlagos pole 1:27.470, 2026 Sepang pole 1:35.130, 2026 Hungary fastest lap 1:22.000 (lap 58), 2025 Las Vegas pole 1:47.934: each confirmed by search hits from the FIA, F1.com, Crash.net, PlanetF1 or Racer (see the reference list at the bottom).
- **Not independently checked:** all other cells. They come from Jolpica only. The fetch tool transcribed the JSON; it is not a hand copy. Treat a value as "database value, not cross-checked" unless it appears in the spot-check list above.

## Barcelona-Catalunya, layout without the final chicane (4.657 km, used by F1 since 2023)

- Wikipedia (https://en.wikipedia.org/wiki/Circuit_de_Barcelona-Catalunya, lap-records table, read 2026-11-16) says: "From the 2023 Spanish Grand Prix, Formula One stopped using the 2007 chicane." It gives the F1 **race lap record** on the "Without Chicane (4.657 km)" layout as **1:15.743, Oscar Piastri, McLaren MCL39, 2025 Spanish GP**. The old with-chicane (4.675 km) record is 1:18.149 (Verstappen, 2021). One search extract says Piastri's 2025 lap beat Verstappen's 2023 record (1:16.330) by 0.587 s, which agrees with Jolpica's 2023 value.
- **Fastest lap ever on this layout (any session, qualifying included)**: **1:11.383, Lando Norris, 2024 Spanish GP pole.** Jolpica gives Q3 = 1:11.383, and F1.com reports Norris beating Verstappen to pole by 0.020 s (https://www.formula1.com/en/latest/article/norris-beats-verstappen-to-pole-in-thrilling-spanish-gp-qualifying-session.60VtCqegJb3U1Vxg4PXfke). This is my derivation from the 2023–2026 poles in the table below; the FIA does not publish an official "qualifying lap record". I did not check practice sessions.
- The 2026 Barcelona event was called the "Barcelona(-Catalunya) Grand Prix". The 2026 "Spanish Grand Prix" was held at the Madring in Madrid.
- The game layout matches. Codemasters put the no-chicane Barcelona layout into F1 23 (reported by racinggames.gg, https://racinggames.gg/f1/f1-23-tracks-full-calendar-new-circuits-alterations-more). I did not find an official patch note for F1 25.

## Table: pole and fastest race lap

Pole = best Q3 time of the pole-sitter in Grand Prix qualifying (sprint qualifying excluded). FL = fastest lap of the Grand Prix (driver, lap number). "n/h" = not held.

| Circuit | Season | Pole | Fastest race lap | Source |
|---|---|---|---|---|
| **Barcelona-Catalunya (4.657 km)** | 2023 (context) | 1:12.272 Verstappen | 1:16.330 Verstappen (L61) | Jolpica circuits/catalunya |
| **Barcelona-Catalunya (4.657 km)** | 2024 (context) | **1:11.383 Norris** (fastest ever on layout) | 1:17.115 Norris (L51) | Jolpica; F1.com 2024 quali article |
| **Barcelona-Catalunya (4.657 km)** | **2025** | **1:11.546 Piastri** | **1:15.743 Piastri (L61), race lap record** | Jolpica; FIA news + FIA FL PDF; Wikipedia record table |
| **Barcelona-Catalunya (4.657 km)** | **2026** | **1:14.679 Russell** | **1:20.122 Hamilton (L44)** | Jolpica; FIA news 13.06.26 (pole); lightsoutblog (FL driver only) |
| Albert Park | 2025 | 1:15.096 Norris | 1:22.167 Norris (L43) | Jolpica |
| Albert Park | 2026 | 1:18.518 Russell | 1:22.091 Verstappen (L43) | Jolpica |
| Shanghai | 2025 | 1:30.641 Piastri | 1:35.454 Norris (L53) | Jolpica |
| Shanghai | 2026 | 1:32.064 Antonelli | 1:35.275 Antonelli (L52) | Jolpica |
| Suzuka | 2025 | 1:26.983 Verstappen | 1:30.965 Antonelli (L50) | Jolpica |
| Suzuka | 2026 | 1:28.778 Antonelli | 1:32.432 Antonelli (L49) | Jolpica |
| Bahrain (Sakhir) | 2025 | 1:29.841 Piastri | 1:35.140 Piastri (L36) | Jolpica |
| Bahrain (Sakhir) | 2026 | n/h (cancelled 15 Mar 2026, Middle East conflict) | n/h | PlanetF1, The National (see refs) |
| *Bahrain GP held at Sepang (~5.5 km, per press)* | 2026 | 1:35.130 Verstappen | 1:38.220 Verstappen (L55) | Jolpica; F1.com / Crash.net quali (pole) |
| Jeddah | 2025 | 1:27.294 Verstappen | 1:31.778 Norris (L41) | Jolpica |
| Jeddah | 2026 | n/h (cancelled, not replaced) | n/h | PlanetF1, The Race (see refs) |
| Miami | 2025 | 1:26.204 Verstappen (Jolpica empty) | 1:29.746 Norris (L36) | Crash.net (pole); Jolpica (FL) |
| Miami | 2026 | 1:27.798 Antonelli | 1:31.869 Norris (L35) | Jolpica |
| Imola | 2025 | 1:14.670 Piastri | 1:17.988 Verstappen (L58) | Jolpica |
| Imola | 2026 | n/h (not on 2026 calendar) | n/h | Jolpica 2026 schedule |
| Monaco | 2025 | 1:09.954 Norris | 1:13.221 Norris (L78) | Jolpica |
| Monaco | 2026 | 1:12.051 Antonelli | 1:13.481 Antonelli (L76) | Jolpica; FIA/F1.com (pole) |
| Montreal | 2025 | 1:10.899 Russell | 1:14.119 Russell (L63) | Jolpica |
| Montreal | 2026 | 1:12.578 Russell | 1:14.210 Antonelli (L68) | Jolpica |
| Red Bull Ring | 2025 | 1:03.971 Norris | 1:07.924 Piastri (L59) | Jolpica |
| Red Bull Ring | 2026 | 1:06.113 Russell | 1:10.374 Antonelli (L59) | Jolpica |
| Silverstone | 2025 | 1:24.892 Verstappen | 1:29.337 Piastri (L51) | Jolpica |
| Silverstone | 2026 | 1:28.111 Antonelli | 1:31.777 Antonelli (L37) | Jolpica |
| Spa | 2025 | 1:40.562 Norris | 1:44.861 Antonelli (L32) | Jolpica |
| Spa | 2026 | 1:44.361 Antonelli | 1:48.890 Norris (L44) | Jolpica |
| Hungaroring | 2025 | 1:15.372 Leclerc | 1:19.409 Russell (L45) | Jolpica |
| Hungaroring | 2026 | 1:17.207 Norris | 1:22.000 Leclerc (L58) | Jolpica; FIA provisional classification (FL) |
| Zandvoort | 2025 | 1:08.662 Piastri | 1:12.271 Piastri (L60) | Jolpica |
| Zandvoort | 2026 | 1:11.163 Norris | 1:14.230 Leclerc (L60) | Jolpica |
| Monza | 2025 | 1:18.792 Verstappen | 1:20.901 Norris (L53) | Jolpica |
| Monza | 2026 | 1:21.786 Gasly | 1:23.504 Antonelli (L53) | Jolpica; FIA/Crash.net (pole) |
| Madring (Madrid) | 2026 | 1:31.824 Norris | 1:35.587 Russell (L49) | Jolpica; McLaren/PlanetF1/Crash.net (pole) |
| Baku | 2025 | 1:41.117 Verstappen | 1:43.388 Verstappen (L50) | Jolpica |
| Baku | 2026 | 1:42.526 Russell | 1:44.916 Russell (L49) | Jolpica |
| Singapore | 2025 | 1:29.158 Russell | 1:33.808 Hamilton (L48) | Jolpica |
| Singapore | 2026 | 1:31.373 Verstappen | 1:35.202 Leclerc (L41) | Jolpica |
| COTA | 2025 | 1:32.510 Verstappen | 1:37.577 Antonelli (L33) | Jolpica |
| COTA | 2026 | 1:34.035 Antonelli | 1:37.976 Leclerc (L42) | Jolpica |
| Mexico City | 2025 | 1:15.586 Norris | 1:20.052 Russell (L50) | Jolpica |
| Mexico City | 2026 | 1:16.514 Hamilton | 1:19.268 Hamilton (L50) | Jolpica |
| Interlagos | 2025 | 1:09.511 Norris | 1:12.400 Albon (L59) | Jolpica |
| Interlagos | 2026 | 1:27.470 Norris (**WET** quali) | 1:12.689 Antonelli (L55) | Jolpica; F1.com/Crash.net (pole, rain) |
| Las Vegas | 2025 | 1:47.934 Norris (**WET** quali) | 1:33.365 Verstappen (L50) | Jolpica; The Race/PlanetF1 (wet) |
| Las Vegas | 2026 | not yet held (race 22 Nov 2026) | not yet held | Jolpica 2026 schedule |
| Lusail | 2025 | 1:19.387 Piastri | 1:22.996 Piastri (L44) | Jolpica |
| Lusail | 2026 | not yet held (race 29 Nov 2026) | not yet held | Jolpica 2026 schedule |
| Yas Marina | 2025 | 1:22.207 Verstappen | 1:26.725 Leclerc (L45) | Jolpica |
| Yas Marina | 2026 | not yet held (race 6 Dec 2026) | not yet held | Jolpica 2026 schedule |

### Caveats on using these as references (my reading)
- **Wet sessions:** the 2025 Las Vegas and 2026 Interlagos poles were set in the wet and are not dry references. I checked the weather only for those two. Any other session could also have been affected by weather; I did not check.
- **Race fastest laps depend on strategy.** In 2026 the gap between race fastest lap and pole is often larger: Barcelona 2026 is +5.4 s (+7.3 %), against +4.2 s (+5.9 %) in 2025. Hamilton's 2026 fastest lap came on lap 44. This fits 2026 energy-management racing, but no source I read states that. Pole laps are the better reference for single-lap pace.
- **2026 cars are slower than 2025 cars on single-lap pace** at most tracks. At Barcelona the pole went 1:11.546 → 1:14.679 (+4.4 %). Compare game times with the matching car generation.

## Game context

### There is no standalone "F1 26" game
- EA announced in Nov 2025 that there would be no F1 26. The 2026 season is a paid expansion for F1 25, and the next full game is due in 2027. Sources: Racer, https://racer.com/2025/11/19/no-f1-26-videogame-ea-sports-says ; GTPlanet, https://www.gtplanet.net/f1-26-expansion-f1-25-20251118/
- EA says the "F1 25: 2026 Season Pack and F1 25: 2026 Season Edition launch worldwide on June 3, 2026 at 3PM UTC". The 2026 cars are available in Time Trial. The Madring is drivable only with 2026 cars. Source: https://careers.ea.com/games/f1/news/f1-25-2026-season-pack-etyntk (an EA page; fetched 2026-11-16).
- F1Laps runs separate "f1-25" and "f1-26" leaderboards. My reading is that "f1-26" means the 2026 Season Pack cars.

### Game Time Trial times compared with real poles (ANECDOTAL)
- I found **no source that documents a systematic comparison** between official EA Time Trial world records and real pole laps. Search results were forum posts only (e.g. EA forums, "F1 23 cars are too fast / not matching real-life times", https://forums.ea.com/discussions/f1-23-en/f1-23-cars-are-too-fast-not-matching-real-life-times/8382715/replies/8382717). These are opinion, not data.
- F1Laps says its all-mode averages "can land faster than real F1 or Esports records" and that "for a fair comparison, Time Trial is the mode to look at" (https://www.f1laps.com/f1-25/average-lap-times/). That page's "F1 Record" for Spain is 1:18.149, the old with-chicane record, so its record column is out of date for Barcelona.
- My own sample, comparing **F1Laps community Time Trial bests** (dry, valid laps with telemetry, from F1Laps app users only, *not* the official EA world records) with the real pole of the matching season. Fetched 2026-11-16:

| Track | Game build | F1Laps TT best | Real pole | Game vs real |
|---|---|---|---|---|
| Barcelona | F1 25 | 1:12.537 | 1:11.546 (2025) | +1.4 % (slower) |
| Suzuka | F1 25 | 1:27.015 | 1:26.983 (2025) | +0.04 % |
| Monaco | F1 25 | 1:10.031 | 1:09.954 (2025) | +0.1 % |
| Red Bull Ring | F1 25 | 1:04.245 | 1:03.971 (2025) | +0.4 % |
| Bahrain | F1 25 | 1:27.881 | 1:29.841 (2025) | −2.2 % (faster) |
| Barcelona | F1 26 (2026 pack) | 1:13.395 | 1:14.679 (2026) | −1.7 % (faster) |
| Suzuka | F1 26 (2026 pack) | 1:28.602 | 1:28.778 (2026) | −0.2 % |
| Monaco | F1 26 (2026 pack) | 1:09.478 (2nd: 1:10.888) | 1:12.051 (2026) | −3.6 % (P2: −1.6 %) |
| Red Bull Ring | F1 26 (2026 pack) | 1:04.312 | 1:06.113 (2026) | −2.7 % |

Leaderboard URLs: https://www.f1laps.com/f1-25/leaderboard/{spain,japan,monaco,austria,bahrain}/ and https://www.f1laps.com/f1-26/leaderboard/{spain,japan,monaco,austria}/.

My reading: the top community TT laps fall within about ±2–4 % of the real pole. For F1 25 (2025 cars) they are roughly equal to or slightly slower than real poles. For the 2026 pack they tend to be faster than the real 2026 poles. Official EA world records are probably faster than F1Laps community bests, so the true WR-vs-pole gap probably leans further toward "game faster". This is unverified and anecdotal.

## Implications for us (interpretation)
1. For Barcelona sanity checks, use **2025 pole 1:11.546** and **2025 fastest race lap 1:15.743** (race lap record) as the 2025-car single-lap and race-pace references. Use **2026 pole 1:14.679 / fastest lap 1:20.122** for 2026-pack cars. The fastest lap ever on the layout is 1:11.383 (Norris, 2024).
2. Treat a game Time Trial lap within about ±2–3 % of the matching-season pole as "realistic". A larger gap more likely points to a setup, car-generation or mode mismatch (My Team/Career cars are faster) than to real driver skill.
3. Do not use the wet poles (2025 Las Vegas, 2026 Interlagos) as references. For 2026 Bahrain/Jeddah/Imola there is no real data at those tracks, and Las Vegas, Lusail and Yas Marina 2026 are still to run.

## References
- Jolpica-F1 API: https://api.jolpi.ca/ergast/f1/2025/qualifying/1.json ; https://api.jolpi.ca/ergast/f1/2025/fastest/1/results.json ; https://api.jolpi.ca/ergast/f1/2026/qualifying/1.json ; https://api.jolpi.ca/ergast/f1/2026/fastest/1/results.json ; https://api.jolpi.ca/ergast/f1/2026.json ; https://api.jolpi.ca/ergast/f1/circuits/catalunya/qualifying/1.json ; https://api.jolpi.ca/ergast/f1/circuits/catalunya/fastest/1/results.json
- FIA 2026 Barcelona quali: https://www.fia.com/news/f1-russell-storms-barcelona-pole-ahead-hamilton-and-antonelli-leclerc-crashes
- FIA 2025 Spain quali: https://api.fia.com/news/f1-piastri-grabs-pole-spain-mclaren-lock-out-front-row-ahead-verstappen
- FIA 2025 Spain race fastest laps PDF: https://api.fia.com/sites/default/files/2025_09_esp_f1_r0_timing_racefastestlaps_v01.pdf (could not extract text myself)
- FIA 2026 Hungary race provisional classification: https://www.fia.com/sites/default/files/2026_11_hun_f1_r0_timing_raceprovisionalclassification_v01.pdf (seen via search extract)
- Wikipedia, Circuit de Barcelona-Catalunya lap records: https://en.wikipedia.org/wiki/Circuit_de_Barcelona-Catalunya
- F1.com 2024 Spain quali: https://www.formula1.com/en/latest/article/norris-beats-verstappen-to-pole-in-thrilling-spanish-gp-qualifying-session.60VtCqegJb3U1Vxg4PXfke
- Lightsoutblog 2026 Barcelona stats: https://www.lightsoutblog.com/2026-barcelona-grand-prix-statistics/
- Crash.net 2025 Miami quali: https://www.crash.net/f1/results/1069184/1/2025-f1-miami-grand-prix-qualifying-results
- 2026 Monaco pole: https://api.fia.com/news/f1-antonelli-takes-monaco-pole-just-ahead-verstappen-and-hamilton-qualifying-thriller
- 2026 Monza pole: https://www.crash.net/f1/results/1103771/1/f1-2026-italian-grand-prix-full-qualifying-result-monza
- 2026 Madring pole: https://www.mclaren.com/racing/formula-1/2026/spanish-grand-prix/qualifying-report/ ; https://www.crash.net/f1/results/1104497/1/f1-2026-spanish-grand-prix-full-qualifying-result-madrid
- 2026 Interlagos wet pole: https://www.crash.net/f1/news/1109818/1/lando-norris-takes-pole-heavily-delayed-wet-brazil-gp-qualifying
- 2026 Sepang (Bahrain GP in Malaysia) pole: https://www.crash.net/f1/results/1106313/1/f1-bahrain-grand-prix-malaysia-full-qualifying-results ; relocation: https://www.formula1.com/en/latest/article/formula-1-and-fia-confirm-formula-1-and-fia-confirm-malaysia-will-join-2026-calendar-as-host-venue-for-the-bahrain-grand-prix.6lL7vjFEM2VVynRHvg1TCf
- 2025 Las Vegas wet pole: https://www.the-race.com/formula-1/norris-beats-verstappen-to-pole-in-wet-las-vegas-qualifying/
- 2026 Bahrain/Saudi cancellation: https://www.planetf1.com/news/bahrain-saudi-arabian-f1-2026-cancelled ; https://www.thenationalnews.com/sport/f1/2026/03/17/f1-bahrain-saudi-arabia-grand-prix-cancelled
- F1 23 Barcelona layout: https://racinggames.gg/f1/f1-23-tracks-full-calendar-new-circuits-alterations-more ; FIA homologation: https://api.fia.com/news/fia-homologates-updated-layout-2023-fia-formula-1-spanish-grand-prix
- No F1 26 / 2026 Season Pack: https://racer.com/2025/11/19/no-f1-26-videogame-ea-sports-says ; https://www.gtplanet.net/f1-26-expansion-f1-25-20251118/ ; https://careers.ea.com/games/f1/news/f1-25-2026-season-pack-etyntk
- F1Laps: https://www.f1laps.com/f1-25/average-lap-times/ and the leaderboard URLs above
