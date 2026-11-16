# F1Laps "meta" setups and Time Trial vs Qualifying rules (EA F1 25 / 2026 Season Pack)

Research note for C2. Compiled 2026-11-16 by the web-researcher. All pages fetched on 2026-11-16 unless stated.
Convention: **Source says** = quoted/paraphrased from the cited page. **My reading** = interpretation. **Anecdotal** = forum/Steam/user post, not EA/Codemasters.

Context on naming: EA announced in Nov 2025 there would be no separate "F1 26" game; the 2026 season ships as a paid expansion ("2026 Season Pack") for F1 25 (VGC, 2025-11; Motorsport.com, 2025-11; https://www.videogameschronicle.com/news/ea-sports-says-therell-be-no-f1-26-with-f1-25-getting-a-paid-2026-season-expansion-instead). EA's deep dive dates the launch 3 June 2026 (https://www.ea.com/en/games/f1/f1-25/news/f1-25-features-deep-dive, dated 2026-05-26). F1Laps labels this content "F1 26" (its supported-games FAQ lists "F1 26 (2026) - Full support" without saying it is the expansion; https://www.f1laps.com/faqs/supported-games/). My reading: f1laps.com/f1-26/ = F1 25 + 2026 Season Pack.

---

## Q1. How does f1laps.com compute its per-track "meta" setup?

### What the site states (all of it)

1. **Selection = the 100 fastest setups per track, ranked by lap time.**
   - Blog "Introducing Meta Setup: Data-Driven Car Settings for Every Track", by "Olivia", F1Laps News, 2026-01-02 (https://www.f1laps.com/blog/introducing-meta-setup-data-driven-car-settings-for-every-track/): "For each track, F1Laps pulls the top 100 setups ranked by lap time and runs the numbers on every parameter." and "Meta Setup aggregates across many fast laps, so individual quirks wash out."
   - F1 26 Australia meta page (https://www.f1laps.com/f1-26/setups/australia/meta/): "From thousands of setups and millions of laps tracked on F1Laps, we analyzed the 100 fastest setups for Australia."
   - F1 25 Australia meta page (https://www.f1laps.com/f1-25/setups/australia/meta/): same sentence, plus "Based on the top 100 fastest setups for Australia."
   - Where fewer setups exist the count is lower or the page says "Not enough setups to generate recommendations" (search-engine snippet for F1 26 Japan; a search snippet for F1 26 Monaco said 82 setups, but the live page fetched today says "100 fastest setups for Monaco", so the pool is refreshed over time; https://www.f1laps.com/f1-26/setups/monaco/meta/).
2. **Dry and wet are computed separately.** Blog: separate recommendations for dry and wet. The meta pages show two blocks under a "Dry / Wet" toggle (the second block is not explicitly labelled in the fetched text).
3. **Range bars and "important" flags.** Meta page: "The bar shows the range used by fast drivers. Narrow bars = most fast drivers use similar values. Wide bars = personal preference." Blog: "Tight clustering suggests consensus." "Parameters marked "important" are the consensus ones."
4. **How setups enter the pool.** FAQ "How to find F1 car setups for wet conditions" (https://www.f1laps.com/faqs/find-wet-weather-setups/): a setup "is automatically shared with the community when you: Set a competitive lap time using the setup; Have the F1Laps Telemetry App capturing your data." "Competitive" is not defined. Track setup list page (https://www.f1laps.com/f1-25/setups/australia/): "The free F1Laps telemetry app records every lap you drive at Australia, in every game mode."
5. **Composition of the top-100 pool (controller).** Setup list pages state, e.g. F1 25 Australia: "Of the 100 fastest dry setups, 61 were driven with a wheel and 18 with a controller; 21 don't say." F1 26 Australia: "45 were driven with a wheel and 18 with a controller; 37 don't say." (https://www.f1laps.com/f1-26/setups/australia/)
6. **Session types in the pool.** The setup list pages offer filters Session = Any / Time Trial / Race / Qualifying, Controller = Any / Controller / Wheel, plus team and assist filters. In the default view, the top 20 rows shown on both F1 25 and F1 26 Australia are all "Time Trial" with 20 distinct users (my count from the fetched page; only 20 rows are rendered, no pagination visible).

### What the site does NOT state (could not verify)

Searched: meta pages (F1 25/F1 26 Australia, F1 26 Monaco), the announcement blog, FAQ index (https://www.f1laps.com/faqs/), "Where can I find car setups" FAQ (https://www.f1laps.com/faqs/find-car-setups/), wet-setup FAQ, MCP FAQ (https://www.f1laps.com/faqs/f1laps-mcp/ — only says "Review your car setup against the community range"). No changelog found. No Reddit/Discord post by the developer found by search (searches returned only f1laps.com pages).
- **Statistic** (median / mean / mode / trimmed): not stated anywhere I could open.
- **Weighting** (by lap time, popularity): not stated. Nothing suggests weighting.
- **Session filter** for the meta (TT only vs all modes): not stated.
- **Controller filter**: not stated; item 5 implies the "100 fastest" pool mixes wheel and pad.
- **Time window / patch version**: not stated (only per-game split: f1-25 vs f1-26).
- **One setup per user (dedup)**: not stated. Top-20 list rows are all distinct users, which is consistent with per-user dedup but does not prove it.
- **Range-bar definition** (min–max vs percentiles): not stated.

### My reading of the evidence on the statistic (interpretation, not a site statement)

The displayed meta values include half-steps on parameters that the same pages otherwise show as whole numbers, e.g. F1 25 Australia rear suspension 3.50, rear ride height 47.50; F1 26 Australia rear wing 23.5, diff off-throttle 42.5 %, brake pressure 99.5 %; F1 26 Monaco front suspension 38.50, rear ARB 6.50, diff 27.50 % / 72.50 %. With N = 100 (even), a per-parameter median is the mean of the 50th and 51st values and produces exactly these x.5 values; an unrounded mean would usually produce arbitrary decimals. So the numbers are **consistent with a per-parameter median over an even-sized pool**, but a mean rounded to 0.5 would also fit. I did not verify the in-game step sizes from a primary source. Also note a per-parameter median produces a "synthetic" setup that no single driver necessarily ran.

---

## Q2. Time Trial (TT) vs Qualifying in EA F1 25 (incl. 2026 Season Pack)

Overall: EA/Codemasters do not publish a mode-by-mode rules table. Official sources only cover track-limit enforcement in TT and tyre-model tuning in TT. Everything else below is a secondary-site or forum claim, labelled as such.

| Topic | Time Trial | Qualifying (career / Grand Prix / online) | Evidence grade |
|---|---|---|---|
| ERS deployment | Contested. Claims range from "unlimited ERS" to "battery refilled each lap" to "locked to Hotlap"; a reply in the same thread contradicts the "locked" claim. | Normal energy rules; manual modes or ERS assist; 2026 pack adds Boost Mode and Overtake Mode (Overtake requires being within 1 s of the car ahead at the detection point). | TT: secondary/anecdotal only. Quali/2026 energy: official. |
| Track limits / invalidation | Own TT ruleset, tightened in patch v1.11 (Aug 2025). Leaderboard reset after a corner-cutting loophole at Madrid was closed (v1.23, per secondary summary). | "Corner Cutting Stringency" setting, Regular (default) or Strict, in Grand Prix/career; F1 23 Ranked was set to Strict (EA tweet). F1 25 Ranked setting not verified. | TT tightening: official. Quali settings: official for F1 22/F1 23, not re-verified for F1 25. |
| Slipstream / tow | No source found that says whether TT ghosts give a tow. My reading: ghosts are replays of other laps, not physical cars, so a tow is unlikely, but this is **not verified**. | Slipstream model active when real cars are present ("slipstream model has been adjusted", F1 25 deep dive). | Unverified for TT. |
| Track evolution / grip | Forum claims (F1 2020, 2020-07): TT "always has perfect conditions" / max grip, with flat track surface grip. Not verified for F1 25. | Not documented for F1 25 by EA. | Anecdotal only. |
| Tyre temperature at lap start | Guide sites say TT "often starts you in range". Not verified with EA text. | Needs an out-lap (general guides). | Secondary/anecdotal only. |
| Fuel load | Guides say TT removes fuel as a variable. EA text not found. | Fuel set by player; v1.27 fixed an F2 bug giving "too much fuel during a Qualifying session" (official). | TT: secondary only. |
| Car performance | F1Laps blog: "Time Trial gives unlimited ERS and equal cars, qualifying is energy-limited with real car performance." | Real team performance / upgrades. | Secondary (F1Laps, 2026-07-31). |

### Details and citations

**Track limits in TT (official).**
- EA, "F1® 25 – v1.11 – Patch Notes", dated 2025-08-15, roll-out 2025-08-18 (https://www.ea.com/games/f1/f1-25/news/f1-25-patch-notes-v111): "Track Limit rules in Time Trial have been tightened to reduce the instances of minor infractions going unpunished".
- EA, v1.13 Patch Notes, dated 2025-10-10, roll-out 2025-10-14 (https://www.ea.com/games/f1/f1-25/news/f1-25-patch-notes-v113): "Fixed an issue which could cause multiple track limits warnings to be shown for a single incident." (not TT-specific).
- v1.23 (2026 Season Pack era), from a secondary summary of the patch notes in a search result (Operation Sports, https://www.operationsports.com/f1-25-update-includes-madrid-track-changes-and-wheel-support-updates-patch-notes): a Madrid corner-cutting loophole was closed, "Because of this change we're also resetting the Madrid Time Trial Leaderboard." I did not open the EA v1.23 page itself.
- Anecdotal (F1 25 bug report, EA forums; could not open, 403, seen only in search snippet): wall-riding at Monaco T5–6 did not invalidate the lap, "only occurs in Time Trial" (https://forums.ea.com/idea/f1-25-bug-reports-en/wall-riding-t5-6-at-monaco-doesnt-invalidate-the-lap/12473016).
- Anecdotal (F1 2020 Steam thread, 2020-07-22): "TT has the strictest cutting rules" (https://steamcommunity.com/app/1080110/discussions/0/2787116918298510847/).

**Corner cutting in Grand Prix / career / online (official, older titles).**
- EA Accessibility resource, F1 22 PC "Rules" (https://ea.com/able/resources/f1-22/pc/rules): "Corner Cutting Stringency … Select an option to specify how strict the corner cutting detection will be. The default is Regular." (Standard and Expert race modes only.)
- Traxion, 2023-08-17 (https://traxion.gg/f1-23s-ranked-track-limits-set-to-be-stricter/), quoting @EASPORTSF1: "Ranked will have corner cutting set to Strict after the next patch." That is F1 23; I found no F1 25 statement.
- Secondary (racinggames.gg, https://racinggames.gg/article/its-time-to-fix-track-limits-in-the-formula-1-franchise): Strict = keep at least one wheel inside the white lines at all times; Regular is more lenient.
- I could not find a primary source that defines the TT ruleset relative to Regular/Strict.

**ERS / energy.**
- Official, 2026 Season Pack: Codemasters designers Steven Embling and Yashar Kiani (EA news, 2026-06-03, https://ea.com/en-gb/news/f1-25-2026-season-pack): Overtake Mode applies "if you're within one second of the car in front", raises the harvest limit, and "The faster you go, the less electrical deployment is available"; assists can automate energy use. Patch v1.25 (via search summary of https://forums.ea.com/blog/f1-games-game-info-hub-en/f1%C2%AE-25-2026-season-pack-%E2%80%93-v1-25-%E2%80%93-patch-notes/13694119, page itself returned 403): Boost Mode available in wet with limited power; Straight Mode and DRS active only with throttle > 90 %. v1.27, dated 2026-11-13 (https://www.ea.com/games/f1/f1-25/news/f1-25-2026-season-pack-patch-notes-v127): "Fixed an issue which could sometimes cause Overtake mode to not be activated despite being one second behind the car ahead at the activation point."
- None of these EA pages say anything about ERS in Time Trial. My reading: Overtake Mode needs a car within 1 s, which a TT ghost probably does not count as, but unverified.
- F1Laps blog "What AI Difficulty Should You Use in F1 26?", Olivia, 2026-07-31 (https://www.f1laps.com/blog/what-ai-difficulty-should-you-use-in-f1-26/): "Time Trial gives unlimited ERS and equal cars, qualifying is energy-limited with real car performance." Also: "Across 4,637 player-and-track pairs on F1 25 … the median player's Time Trial lap was 0.65% slower than their own qualifying lap." This is a third-party statement, not EA's.
- Anecdotal, contradictory: Steam F1 25 thread "ERS in Time trial" (Jan, https://steamcommunity.com/app/3059520/discussions/0/742664495215769136/): one user "I believe … ERS is generally locked to its maximum "Hotlap" setting by default"; the original poster replies that a video shows ERS used in TT "like in race". Guide sites (f125game.com, not an established outlet) say "Time Trial refills battery at the line". F1 2020 Steam thread: TT cars run "at full power all the time".
- Not opened: overtake.gg F1 25 ERS guide (403).

**Slipstream / ghosts.**
- EA F1 25 deep dive, 2025-04-17 (https://www.ea.com/games/f1/f1-25/news/f1-25-advancements-deep-dive): "our slipstream model has been adjusted for more improved racing." No TT mention.
- EA Accessibility, F1 22 Time Trial (https://ea.com/able/resources/f1-22/pc/time-trial): ghost options ("Display My Ghost", default On; "Display Rivals Ghost", default Off). No statement on tow. The F1 25 equivalent URL returned 404.
- A Codemasters helpshift FAQ (https://codemasters.helpshift.com/hc/faq/600-what-is-the-slipstream-effect-1658498230) says slipstream is not available in "Qualifying races" because the rival car is transparent; my reading is that this is the F1 Mobile/Clash-type product, not EA F1 25, so do not use it.
- Result: **no source found** on tow in F1 25 TT.

**Track evolution / grip.** No EA source. Anecdotal only (F1 2020 Steam thread above): "Time Trial always has perfect conditions from track grip to the cars weight and tyre temperatures"; another user: surface grip stays flat in TT and rises in GP. racinggames.gg beginners guide: TT "Conditions are always perfect" (secondary).

**Tyres.** Official: F1 25 deep dive: "adjustments to tyre loads and the tyre slip model, particularly in Time Trial." (So the tyre model was tuned specifically for TT; how it differs is not stated.) Start-of-lap temperatures: guide-level claims only ("Time Trial often starts you in range"); F1 2020 forum claim of "perfect … tyre temperatures". Not verified.

**Fuel.** No EA source for TT fuel. Guide-level claims that TT removes fuel as a variable. Qualifying fuel is player-set (v1.27 F2 qualifying-fuel bug fix confirms fuel is a quali variable).

---

## Implications for us (interpretation)

1. Treat the F1Laps meta as a **per-parameter robust central value (probably median) of a mixed pool** of the 100 fastest setups per game and track, dominated by Time Trial laps, mixing wheel and pad, with unknown assists and patch versions. It is a "synthetic consensus", not a setup any one driver necessarily ran. If C2 uses it as a target or anchor, record the fetch date; the pool changes over time.
2. If we need the statistic exactly (median vs mean), the only route is to ask F1Laps or compute it ourselves from the listed setups (filter Session = Time Trial, Dry). The F1Laps MCP connector exposes "community range" data and may let us do that, with the user's approval.
3. TT laps (which make up most of the meta pool and leaderboards) should be modelled as: no traffic/tow, best-case grip, low fuel, and probably more generous energy use than qualifying. All of this is unverified apart from TT-specific track-limit tuning. Do not treat TT lap times as equivalent to qualifying times; F1Laps' own data puts the median player's TT lap 0.65 % slower than their quali lap (that mixes skill and familiarity effects, so it is not a pure rules effect).
4. For the 2026 pack, the energy model (Boost/Overtake/Straight Mode) differs from F1 25's 2025 rules, so f1-25 and f1-26 meta pools should not be pooled.
5. Open items needing in-game checks (UDP telemetry would settle most): fuel mass at TT lap start, tyre carcass/surface temps at TT lap start, ERS store at lap start and whether it resets per lap, track grip/temperature fields per session type. The F1 25 UDP `CarStatus` and `Session` packets carry fuel, ERS store, tyre temps and track temperature, so a short TT vs quali capture would answer these directly.

## Sources
- https://www.f1laps.com/blog/introducing-meta-setup-data-driven-car-settings-for-every-track/
- https://www.f1laps.com/f1-26/setups/australia/meta/
- https://www.f1laps.com/f1-25/setups/australia/meta/
- https://www.f1laps.com/f1-26/setups/monaco/meta/
- https://www.f1laps.com/f1-25/setups/australia/
- https://www.f1laps.com/f1-26/setups/australia/
- https://www.f1laps.com/faqs/ , https://www.f1laps.com/faqs/find-car-setups/ , https://www.f1laps.com/faqs/find-wet-weather-setups/ , https://www.f1laps.com/faqs/f1laps-mcp/ , https://www.f1laps.com/faqs/supported-games/
- https://www.f1laps.com/blog/what-ai-difficulty-should-you-use-in-f1-26/
- https://www.ea.com/games/f1/f1-25/news/f1-25-patch-notes-v111
- https://www.ea.com/games/f1/f1-25/news/f1-25-patch-notes-v113
- https://www.ea.com/games/f1/f1-25/news/f1-25-2026-season-pack-patch-notes-v127
- https://www.ea.com/games/f1/f1-25/news/f1-25-advancements-deep-dive
- https://www.ea.com/en/games/f1/f1-25/news/f1-25-features-deep-dive
- https://ea.com/en-gb/news/f1-25-2026-season-pack
- https://www.ea.com/games/f1/f1-25/news/f1-25-2026-season-edition-tips-and-tricks
- https://ea.com/able/resources/f1-22/pc/rules , https://ea.com/able/resources/f1-22/pc/time-trial
- https://traxion.gg/f1-23s-ranked-track-limits-set-to-be-stricter/
- https://traxion.gg/how-f1-25s-2026-season-pack-replicates-the-new-regulations/
- https://www.operationsports.com/f1-25-update-includes-madrid-track-changes-and-wheel-support-updates-patch-notes (search-snippet only)
- https://racinggames.gg/article/its-time-to-fix-track-limits-in-the-formula-1-franchise (search-snippet only)
- https://steamcommunity.com/app/3059520/discussions/0/742664495215769136/ (anecdotal)
- https://steamcommunity.com/app/1080110/discussions/0/2787116918298510847/ (anecdotal, F1 2020)
- https://www.videogameschronicle.com/news/ea-sports-says-therell-be-no-f1-26-with-f1-25-getting-a-paid-2026-season-expansion-instead

Could not open (403/404): overtake.gg F1 25 ERS guide; EA forums (community-raised-issues tracker, bug reports, Italian "ERS nelle prove a tempo" thread, v1.25 patch-notes blog); EA Accessibility F1 25 Time Trial page (404).
