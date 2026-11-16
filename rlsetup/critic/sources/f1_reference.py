"""Real-world reference laps for the C2b_addendum B1 real-world check of the F1 game labels.

T_ref = real Grand Prix pole (Q3 time of the pole-sitter) of the season that matches the game's cars:
'F1 25' -> 2025 season, 'F1 26' (= F1 25 with the 2026 Season Pack) -> 2026 season. Values from
docs/research/c2_real_lap_times.md (Jolpica-F1 API, spot-checked against FIA / F1.com / Crash.net there;
Miami 2025 from Crash.net because Jolpica has empty Q times). Seconds.

Left out on purpose (no dry season-matched reference, research note "Caveats"):
- wet poles: Las Vegas 2025, Interlagos 2026;
- not held in 2026: Bahrain (moved to Sepang, a different circuit), Jeddah, Imola;
- not yet held at the research date: Las Vegas, Lusail and Yas Marina 2026;
- Madring (madrid) has no 2025 race.
Weather was checked only for the two wet poles above (research note). A track without an entry is
skipped by the check (counted, not dropped).
"""

REAL_POLE_S = {
    ('F1 25', 'abudhabi'): 82.207, ('F1 25', 'australia'): 75.096, ('F1 25', 'austria'): 63.971,
    ('F1 25', 'azerbaijan'): 101.117, ('F1 25', 'bahrain'): 89.841, ('F1 25', 'brazil'): 69.511,
    ('F1 25', 'canada'): 70.899, ('F1 25', 'china'): 90.641, ('F1 25', 'hungary'): 75.372,
    ('F1 25', 'imola'): 74.670, ('F1 25', 'japan'): 86.983, ('F1 25', 'mexico'): 75.586,
    ('F1 25', 'miami'): 86.204, ('F1 25', 'monaco'): 69.954, ('F1 25', 'monza'): 78.792,
    ('F1 25', 'netherlands'): 68.662, ('F1 25', 'qatar'): 79.387, ('F1 25', 'saudi_arabia'): 87.294,
    ('F1 25', 'silverstone'): 84.892, ('F1 25', 'singapore'): 89.158, ('F1 25', 'spa'): 100.562,
    ('F1 25', 'spain'): 71.546, ('F1 25', 'usa'): 92.510,
    ('F1 26', 'australia'): 78.518, ('F1 26', 'austria'): 66.113, ('F1 26', 'azerbaijan'): 102.526,
    ('F1 26', 'canada'): 72.578, ('F1 26', 'china'): 92.064, ('F1 26', 'hungary'): 77.207,
    ('F1 26', 'japan'): 88.778, ('F1 26', 'madrid'): 91.824, ('F1 26', 'mexico'): 76.514,
    ('F1 26', 'miami'): 87.798, ('F1 26', 'monaco'): 72.051, ('F1 26', 'monza'): 81.786,
    ('F1 26', 'netherlands'): 71.163, ('F1 26', 'silverstone'): 88.111, ('F1 26', 'singapore'): 91.373,
    ('F1 26', 'spa'): 104.361, ('F1 26', 'spain'): 74.679, ('F1 26', 'usa'): 94.035,
}
SOURCE = 'docs/research/c2_real_lap_times.md (16 Nov 2026)'
