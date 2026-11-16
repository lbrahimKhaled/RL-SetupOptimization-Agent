#!/usr/bin/env python
"""C2 data build: setups -> rows A / S, telemetry laps -> style numbers, R2 geometry gate, track z.

C2b changes (C2b_addendum B20): Mexico dropped at load (B2); label = % gap to the (game, track, session)
median after the two-sided two-pass robust filter and the real-world hard cut (B1), with the real-world
check table; z for Imola / Las Vegas from the relaxed bar, flagged `z_provisional` (B3); a cross-game
integer `person` id per row (B6) and the data hash the folds are keyed to.

Reads every f1laps export under `--sources` in place (folders, zip archives, bare setups CSVs; the
default is everything in setup_data/), prints the row-count log, the geometry-gate table, the style
summary (median, p1, p99 per number), the comparison with C2_review's Evidence and two report-only
tables for the architect (rows without track z, the fast tail of `gap`), and caches the result under
runs/critic_c2_data/<UTC>/ (gitignored): dataset.npz (arrays only) + summary.json (counts, gate
table, style summary, provenance). No model is fitted (C2b). No username is read into any output:
drivers are integer ids, exports are labelled by file/folder name.

Usage: .venv/bin/python scripts/critic_c2_data.py [--sources setup_data/*] [--out runs/critic_c2_data]
       [--no-cache]
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import datetime
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlsetup import runs                                                            # noqa: E402
from rlsetup.critic.config import Config, KNOB_ORDER, LAP_NO_GRAPH, OK, REPO_ROOT, TRACK_FEATURES  # noqa: E402
from rlsetup.critic.dataset import ranking_stats                                     # noqa: E402
from rlsetup.critic.geometry import scale_scan                                       # noqa: E402
from rlsetup.critic.provenance import car_constants_provenance, library_versions     # noqa: E402
from rlsetup.critic.sources.f1_game_map import TRACK_GEOMETRY, F1GameMapConfig       # noqa: E402
from rlsetup.critic.sources.f1laps import (F1LapsAdapter, build_frame, expand_sources, load_exports,  # noqa: E402
                                           real_world_check, s_rows_before_filters)
from rlsetup.critic.folds import data_hash_of                                        # noqa: E402
from rlsetup.critic.style_data import cell_styles, lap_styles, style_summary, track_z_table  # noqa: E402
from rlsetup.critic.track_features import track_features                             # noqa: E402
from rlsetup.provenance import run_provenance                                       # noqa: E402
from rlsetup.style import KEYS, STYLE_RL_COMMIT, STYLE_RL_SOURCES_SHA256              # noqa: E402

# C2b_addendum expectations (E1, E2, E4; 16 Nov), for the side-by-side print
EXPECTED = {'rows passing the box (no label filter, no Mexico)': 20706, 'rows A': 20192, 'dropped low': 71,
            'dropped high': 443, 'exact rows': 586, 'exact rows with style': 579, 'rows S': 1634,
            'Barcelona rows A': 977, 'Barcelona exact rows': 24}
FAST_TAIL_PCT = -4.             # report-only threshold of the fast-tail table (lead's addition b)


def gate_rows(geoms):
    rows = []
    for t in sorted(geoms):
        tg = geoms[t]
        if not tg.results:
            rows.append({'track': t, 'stem': tg.stem, 'kind': '-', 'chosen': False, 'reason': tg.reason})
        for g, r in tg.results:
            rows.append({'track': t, 'chosen': tg.chosen is not None and tg.chosen[0] is g, 'line_kind': g.line_kind,
                         'width_assumed': g.width_assumed, 'file': g.path.name, **r.summary()})
    return rows


def print_gate(rows):
    print('\n== R2 geometry gate (bar: median |corr| >= 0.85, |dL| <= 1 %, median signed corr < 0 for F1 steer; '
          'extra = bacinger, tum = TUM) ==')
    print('closure = scale * L_game - L_geo (where the lap end lands; QA I1); per game = median |corr| per game (QA I2)')
    print(f"{'game track':13s} {'stem':11s} {'cand':5s} {'laps':>4s} {'L_game':>7s} {'L_geo':>7s} {'dL %':>7s} "
          f"{'scale':>7s} {'shift m':>7s} {'closure':>7s} {'|corr|':>6s} {'@s=1':>6s} {'signed':>7s}  per game; result")
    for r in rows:
        if r['kind'] == '-':
            print(f"{r['track']:13s} {r['stem']:11s} {'-':5s} no geometry file")
            continue
        flag = 'PASS' + (' (used)' if r['chosen'] else '') if r['passed'] else 'FAIL ' + ','.join(
            n for n, ok in (('corr', r['corr_ok']), ('sign', r['sign_ok']), ('length', r['length_ok'])) if not ok)
        per = ' '.join(f'{g[-2:]} {v:.3f}' for g, v in r['group_median_abs_corr'].items())
        print(f"{r['track']:13s} {r['stem']:11s} {r['kind']:5s} {r['n_laps']:4d} {r['L_game_m']:7.0f} {r['L_geo_m']:7.0f} "
              f"{100 * r['length_rel']:+7.3f} {r['scale']:7.4f} {r['shift_m']:+7.0f} {r['closure_gap_m']:+7.1f} "
              f"{r['median_abs_corr']:6.3f} {r['shift_only'].get('median_abs_corr', np.nan):6.3f} "
              f"{r['median_signed_corr']:+7.3f}  {per}; {flag}")


def ungated_z_report(frame, geoms, z, cfg):
    """Lead's addition (a), report only: rows A / S on tracks without z, and the z the best failing
    candidate geometry would give, next to the gated tracks' range."""
    gated = np.isfinite(z).all(axis=1)
    lo, hi = np.nanmin(z[gated], axis=0), np.nanmax(z[gated], axis=0)
    no_z = ~gated[frame.track]
    out = {'nan_entries': int(np.isnan(z).sum()), 'tracks_without_z': [t for t, g in zip(frame.track_names, gated) if not g],
           'rows_A_without_z': int(no_z.sum()), 'rows_S_without_z': int((no_z & frame.in_s).sum()),
           'gated_range': {n: [float(a), float(b)] for n, a, b in zip(TRACK_FEATURES, lo, hi)}, 'candidates': {}}
    print(f"\n== track z for the tracks without gated geometry (report only, nothing used) ==")
    print(f"NaN entries in z: {out['nan_entries']}; rows A without z: {out['rows_A_without_z']} of {len(frame)}; "
          f"rows S without z: {out['rows_S_without_z']} of {int(frame.in_s.sum())}")
    print(f"{'':34s} " + ' '.join(f'{n:>18s}' for n in TRACK_FEATURES))
    print(f"{'gated tracks min':34s} " + ' '.join(f'{v:18.3f}' for v in lo))
    print(f"{'gated tracks max':34s} " + ' '.join(f'{v:18.3f}' for v in hi))
    for t in out['tracks_without_z']:
        rows = int((frame.track == frame.track_names.index(t)).sum())
        for g, r in geoms[t].results:
            zz = track_features(g.xy, cfg.track)
            inside = [bool(a - 1e-9 <= v <= b + 1e-9) for v, a, b in zip(zz, lo, hi)]
            out['candidates'][f'{t}:{g.kind}'] = {'z': zz.tolist(), 'inside_gated_range': inside, 'reason': r.reason,
                                                 'rows_A': rows, 'rows_S': int((frame.in_s & (frame.track == frame.track_names.index(t))).sum())}
            print(f"{(t + ' ' + g.kind + ' (' + r.reason + ')'):34s} " + ' '.join(
                f"{v:17.3f}{' ' if ok else '*'}" for v, ok in zip(zz, inside)))
    print('* = outside the gated tracks\' range')
    return out


def fast_tail_report(frame, threshold=FAST_TAIL_PCT):
    """Lead's addition (b), report only: distribution of the fast tail of `gap` (rows below threshold)."""
    q = np.percentile(frame.gap, [0, .1, .5, 1, 5, 50, 95, 99, 99.9, 100])
    tail = frame.gap < threshold
    out = {'threshold_pct': threshold, 'quantiles': dict(zip(['min', 'p0.1', 'p0.5', 'p1', 'p5', 'p50', 'p95', 'p99',
                                                              'p99.9', 'max'], map(float, q))),
           'rows_below': int(tail.sum()), 'rows_below_by_cut': {f'{c:g}': int((frame.gap < c).sum()) for c in (-3, -4, -5, -6, -7, -8)},
           'by_game': {}, 'by_session': {}, 'by_game_track_session': []}
    print(f'\n== fast tail of gap (report only; the 8 % filter is one-sided) ==')
    print('gap quantiles %: ' + ', '.join(f'{k} {v:+.2f}' for k, v in out['quantiles'].items()))
    print('rows below cut: ' + ', '.join(f'{k} %: {v}' for k, v in out['rows_below_by_cut'].items()))
    for gi, g in enumerate(frame.games):
        out['by_game'][g] = int((tail & (frame.game == gi)).sum())
    for si, s in enumerate(frame.sessions):
        out['by_session'][s] = int((tail & (frame.session == si)).sum())
    print(f'rows below {threshold:g} %: {out["rows_below"]} of {len(frame)}; by game {out["by_game"]}; by session {out["by_session"]}')
    keys = {}
    for i in np.flatnonzero(tail):
        keys.setdefault((frame.game[i], frame.track[i], frame.session[i]), []).append(i)
    print(f"{'game':6s} {'track':13s} {'session':7s} {'rows':>4s} {'of':>5s} {'min gap':>8s} {'drivers':>7s}")
    for (g, t, s), ii in sorted(keys.items(), key=lambda kv: -len(kv[1])):
        n_all = int(((frame.game == g) & (frame.track == t) & (frame.session == s)).sum())
        row = {'game': frame.games[g], 'track': frame.track_names[t], 'session': frame.sessions[s], 'rows': len(ii),
               'rows_in_group': n_all, 'min_gap_pct': float(frame.gap[ii].min()), 'drivers': len(set(frame.driver[ii]))}
        out['by_game_track_session'].append(row)
        print(f"{row['game']:6s} {row['track']:13s} {row['session']:7s} {row['rows']:4d} {n_all:5d} "
              f"{row['min_gap_pct']:+8.2f} {row['drivers']:7d}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--sources', nargs='+', default=[str(REPO_ROOT / 'setup_data' / '*')])
    ap.add_argument('--out', default=str(REPO_ROOT / 'runs/critic_c2_data'))
    ap.add_argument('--no-cache', action='store_true', help='recompute every lap; the style cache is neither read nor written')
    args = ap.parse_args(argv)
    cfg, map_cfg = Config(), F1GameMapConfig()
    adapter = F1LapsAdapter(map_cfg, cfg)

    print('== setups and laps ==')
    data = load_exports(expand_sources(args.sources), drop_tracks=cfg.data.drop_tracks, games=cfg.data.games)
    frame = build_frame(data, cfg, map_cfg, adapter=adapter)
    s_before = s_rows_before_filters(data)
    print(f'rows S before filters (all unique setup_id): {s_before}')
    rank_a, rank_s = ranking_stats(frame), ranking_stats(frame, frame.in_s)
    print(f'ranking groups (cell x session, R6), rows A: {rank_a}')
    print(f'ranking groups, rows S: {rank_s}')

    print('\n== style (S3, R2-R4) ==')
    laps, geoms = lap_styles(data, TRACK_GEOMETRY, cfg, adapter=adapter, use_cache=not args.no_cache)
    cells = cell_styles(frame, laps)
    s_rows = frame.in_s
    s_styled = s_rows & np.isin(frame.cell, np.flatnonzero(cells.has_style))
    gates = gate_rows(geoms)
    print_gate(gates)
    for t, tg in sorted(geoms.items()):                     # report only: free scale scan where the length fails
        lk = laps.keys[(laps.track == t) & laps.ok]
        for g, r in tg.results:
            if r.corr_ok and not r.length_ok:
                raw = [(si.raw_distance_m, si.raw_steer_deg) for si in
                       (adapter.style_inputs(data.laps[int(k)], cfg.style_input) for k in lk)]
                best = max(scale_scan(g, raw, np.arange(.90, 1.10001, .0025)), key=lambda x: x[2])
                print(f'  {t} {g.kind}: free scale scan 0.90-1.10 peaks at scale {best[0]:.4f} (|corr| {best[2]:.3f}, '
                      f'shift {best[1]:+.0f} m); closed-lap ratio L_geo/L_game {r.L_geo / r.L_game:.4f}; '
                      f'report only, gate not relaxed')
    print(f'\ntracks with gated geometry: {sum(tg.chosen is not None for tg in geoms.values())} of {len(geoms)}; '
          f"without: { {t: tg.reason for t, tg in geoms.items() if tg.chosen is None} }")
    with_graph = laps.lap_reason != LAP_NO_GRAPH
    ties = laps.n_dropped_ties[with_graph]
    lap_counts = {'laps (all)': len(laps.keys), 'laps without graph samples': int((~with_graph).sum()),
                  'laps with graph samples': int(with_graph.sum()), 'laps with numbers': int(laps.ok.sum()),
                  'rejected by reason (laps with graph samples)': {
                      r: int((laps.lap_reason[with_graph] == r).sum()) for r in sorted(set(laps.lap_reason[with_graph]) - {OK})},
                  'laps with geometry numbers': int((laps.geometry_reason == OK).sum()),
                  'laps by geometry reason': {r: int((laps.geometry_reason[laps.ok] == r).sum())
                                              for r in sorted(set(laps.geometry_reason[laps.ok]))},
                  'cells': len(frame.cells), 'cells with style': int(cells.has_style.sum()),
                  'rows S': int(s_rows.sum()), 'rows S with style': int(s_styled.sum()),
                  'rows S with geometry numbers': int((s_rows & np.isin(frame.cell, np.flatnonzero(cells.geometry_reason == OK))).sum()),
                  'steer sign (corr < 0 = steer + is a right turn)': {'negative': int(np.sum(laps.steer_kappa_corr < 0)),
                                                                      'positive': int(np.sum(laps.steer_kappa_corr > 0))},
                  'distance ties dropped (steer samples; QA M5)': {'laps with ties': int((ties > 0).sum()),
                                                                   'samples': int(ties.sum()), 'max per lap': int(ties.max())}}
    for k, v in lap_counts.items():
        print(f'{k}: {v}')
    summ_laps, summ_cells = style_summary(laps.values[laps.ok]), style_summary(cells.values[cells.has_style])
    print('\n== style numbers over laps with numbers (style-rl units; steer x 115) ==')
    print(f"{'number':26s} {'n':>4s} {'median':>8s} {'p1':>8s} {'p99':>8s}   cells: median [p1, p99]")
    for k in KEYS:
        a, c = summ_laps[k], summ_cells[k]
        print(f"{k:26s} {a['n']:4d} {a['median']:8.3g} {a['p1']:8.3g} {a['p99']:8.3g}   "
              f"{c['median']:.3g} [{c['p1']:.3g}, {c['p99']:.3g}]")
    z, z_prov, z_source = track_z_table(frame.track_names, geoms, cfg)
    print('\n== track features z (raw; B3: provisional = relaxed bar, NaN = none) ==')
    print(f"{'track':13s} " + ' '.join(f'{n:>18s}' for n in TRACK_FEATURES) + '  source')
    for t, row, src in zip(frame.track_names, z, z_source):
        print(f'{t:13s} ' + ' '.join(f'{v:18.3f}' for v in row) + f'  {src}')
    ungated = ungated_z_report(frame, geoms, z, cfg)
    tail = fast_tail_report(frame)
    real = real_world_check(data, cfg)
    print('\n== B1 real-world check: r = fastest f1laps TT top-10 lap / season-matched real pole (review only) ==')
    print(f"{'game':6s} {'track':13s} {'TT best s':>9s} {'T_ref s':>8s} {'r':>6s}")
    for r in real['rows']:
        print(f"{r['game']:6s} {r['track']:13s} {r['fastest_tt_s']:9.3f} {r['T_ref_s']:8.3f} {r['r']:6.3f}"
              f"{'  FLAG (outside [%g, %g])' % cfg.data.real_ref_flag_band if r['flag'] else ''}")
    print('skipped (no dry season-matched reference): ' + ', '.join(f"{x['game']} {x['track']}" for x in real['skipped']))
    hard = frame.box_report['label']['hard_cut']
    print(f"hard cut (lap < {cfg.data.real_ref_min_ratio:g} T_ref): {hard['n_cut']} rows; "
          f"{hard['rows_without_reference']} rows have no reference")

    ex = frame.exact_lap >= 0
    styled_cells = np.flatnonzero(cells.has_style)
    sp = frame.track_names.index('spain') if 'spain' in frame.track_names else -1
    lab = frame.box_report['label']
    measured = {'rows passing the box (no label filter, no Mexico)': int(dict(frame.counts)['3. inside the OCD box']),
                'rows A': len(frame), 'dropped low': sum(p['dropped_low'] for p in lab['passes']),
                'dropped high': sum(p['dropped_high'] for p in lab['passes']), 'exact rows': int(ex.sum()),
                'exact rows with style': int((ex & np.isin(frame.cell, styled_cells)).sum()),
                'rows S': int(s_rows.sum()), 'Barcelona rows A': int((frame.track == sp).sum()),
                'Barcelona exact rows': int((ex & (frame.track == sp)).sum())}
    print('\n== measured vs C2b_addendum expectations ==')
    for k, v in EXPECTED.items():
        print(f"{k:52s} expected {v:>6}  measured {measured[k]:>6}  {'same' if v == measured[k] else 'DIFFERENT'}")
    print(f"label filter dropped by session (sign of the final gap): {lab['dropped_by_session']}")
    both = int(sum(len(set(frame.game[frame.person == q])) > 1 for q in np.unique(frame.person)))
    print(f'persons (cross-game, B6): {len(np.unique(frame.person))} in rows A; with rows in both games: {both}')
    dhash = data_hash_of(data.provenance.files)
    print(f'data hash (B6, folds are keyed to it): {dhash}')

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%SZ')
    out = Path(args.out) / stamp
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'dataset.npz', knob_order=np.array(KNOB_ORDER), knobs=frame.knobs, p=frame.p, gap=frame.gap,
                        lap_s=frame.lap_s, game=frame.game, track=frame.track, driver=frame.driver, team=frame.team,
                        session=frame.session, controller=frame.controller, date_days=frame.date_days,
                        setup_hash=frame.setup_hash.astype(str), cell=frame.cell, exact_lap=frame.exact_lap,
                        person=frame.person, label_median_s=frame.label_median_s,
                        games=np.array(frame.games), track_names=np.array(frame.track_names),
                        team_names=np.array(frame.team_names), sessions=np.array(frame.sessions),
                        controllers=np.array(frame.controllers), cells=np.array(frame.cells, int).reshape(-1, 3),
                        style_keys=np.array(KEYS), cell_style=cells.values, cell_style_n_laps=cells.n_laps,
                        cell_style_reason=cells.reason.astype(str), cell_geometry_reason=cells.geometry_reason.astype(str),
                        lap_keys=laps.keys, lap_game=laps.game.astype(str), lap_track=laps.track.astype(str),
                        lap_driver=laps.driver, lap_style=laps.values, lap_reason=laps.lap_reason.astype(str),
                        lap_geometry_reason=laps.geometry_reason.astype(str), lap_is_placeholder=laps.is_placeholder,
                        lap_line_kind=laps.line_kind.astype(str), lap_median_spacing_m=laps.median_spacing_m,
                        lap_time_closure_rel=laps.time_closure_rel, lap_steer_kappa_corr=laps.steer_kappa_corr,
                        lap_n_dropped_ties=laps.n_dropped_ties,
                        track_features=z, track_feature_names=np.array(TRACK_FEATURES), z_provisional=z_prov,
                        z_source=np.array(z_source), assist_names=np.array(list(frame.assists)),
                        **{f'assist_{a}': v for a, v in frame.assists.items()})
    summary = {'created_utc': runs.utc_now(), 'argv': sys.argv[1:], 'counts': frame.counts, 'box_report': frame.box_report,
               's_before_filters': s_before, 'ranking_A': rank_a, 'ranking_S': rank_s, 'laps': lap_counts,
               'measured_vs_addendum': {k: {'expected': EXPECTED[k], 'measured': measured[k]} for k in EXPECTED},
               'label_filter': lab, 'real_world_check': real, 'dropped_tracks': data.dropped_tracks, 'data_hash': dhash,
               'z_provisional': {t: bool(v) for t, v in zip(frame.track_names, z_prov)},
               'gate': gates, 'style_summary_laps': summ_laps, 'style_summary_cells': summ_cells,
               'ungated_track_z': ungated, 'gap_fast_tail': tail,
               'vocabularies': {'sessions': frame.sessions, 'controllers': frame.controllers,
                                'assist_levels': frame.assist_levels},
               'config': asdict(cfg), 'map_config': asdict(map_cfg),
               'provenance': {'input_files_sha256': data.provenance.files,
                              'duplicate_files_skipped': data.provenance.skipped_duplicates,
                              'style_rl_commit': STYLE_RL_COMMIT, 'style_rl_sources_sha256': STYLE_RL_SOURCES_SHA256,
                              'geometry_sha256': {r['stem'] + ':' + r['kind']: r.get('sha256') for r in gates if r['kind'] != '-'},
                              'placeholder_line_sha256': {r['stem'] + ':' + r['kind']: r.get('line_sha256')
                                                          for r in gates if r['kind'] != '-'},
                              'libraries': library_versions(), 'car_constants': car_constants_provenance(cfg.car),
                              **run_provenance()}}
    (out / 'summary.json').write_text(runs.to_json(summary, indent=1) + '\n')
    print(f'\nwrote {out}/dataset.npz and summary.json')
    return out


if __name__ == '__main__':
    main()
