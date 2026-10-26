# Limebeer-group optimal-control lap times (web-researcher, 26 Oct 2026)

| Paper | Track | Car / solver | Baseline | Optimised | Gain |
|---|---|---|---|---|---|
| Perantoni & Limebeer, VSD 2014 (ORA manuscript, §4.3–4.4, Tables 1–2) | Barcelona-Catalunya | 3-DOF F1, 660 kg, Cl 3.0, Cd 0.9; ICLOCS + IPOPT | 82.43 s (mesh-converged 82.57 s) | 80.94 s (mass centre, centre of pressure, roll balance, diff jointly) | ≈1.5 s. One at a time: CoM 82.425, CoP 82.212, roll balance 81.947, locked diff 81.353 s; mass ≈0.035 s/kg |
| Masouleh & Limebeer, IEEE TCST 2016 (ORA manuscript, §V, Table III) | Barcelona-Catalunya (2D) | F1 + polynomial multi-link suspension + aero map; GPOPS-II | 84.55 s (with suspension) | 82.75 s (all parameters) | "a significant 1.8 s". Ride-height offsets 84.05; fixed aero balance 83.66; variable aero balance 82.49; rigid 83.10 s; table/text conflict 83.54 vs 83.45 s |
| Leyton-Romero & Limebeer, VSD 2026, DOI 10.1080/00423114.2026.2674200 (open access CC BY-NC-ND) | NOT VERIFIED (T&F returned 403) | — | — | — | — |

No Limebeer-group Shanghai lap found (Perantoni 2013 / Masouleh 2017 theses: Barcelona; Höppke 2022 thesis not loadable, supervised by Nakatsukasa & Reisinger; Tremlett not checked).

Sources: https://ora.ox.ac.uk/objects/uuid:856dd2b0-3df9-4daa-a4f7-9462dc7cd2f8/files/m3fb7226495d64be875ddfdc868095771 · https://ora.ox.ac.uk/objects/uuid:ce1a7106-0a2c-41af-8449-41541220809f/files/m776aa23411ad9d78c36f96620a0e0f0b · https://api.crossref.org/works/10.1080/00423114.2026.2674200 · https://ora.ox.ac.uk/objects/uuid:491a5bb1-db1b-4cf6-b6f2-0ec06097ac9d · https://ora.ox.ac.uk/objects/uuid:b75fa69c-ebe0-4812-98e9-984d93476d37 · https://ora.ox.ac.uk/objects/uuid:c55c3e9c-270c-4d47-b5f8-e86621608b24

Interpretation (not from the papers): published setup gains are ~1.5–1.8 s on an ~82–85 s lap (~2 %).
