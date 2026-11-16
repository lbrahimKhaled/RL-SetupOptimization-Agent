"""Stage C2: style-conditioned setup critics for the future TD3 setup agent (replaces C1).

C2a (this code): data sources and adapters (`sources/`), the 15 OCD knobs (`knobs.py`), track geometry
and the R2 gate (`geometry.py`), telemetry lap -> style numbers, cells and the R4 transform
(`style_data.py`), track features z (`track_features.py`).
C2b: data container (`c2data.py`), folds (`folds.py`), B4 style transform (`style_transform.py`),
nuisance (`nuisance.py`), knob groups (`groups.py`), critics (`model.py`), training (`train.py`),
baselines (`ridge.py`), metrics (`metrics.py`, `evaluate.py`), the saved critic (`bundle.py`), settings
and bars (`c2_config.py`); the runs are scripts/critic_c2.py.
Specs: docs/specs/C2_style_critic.md amended by docs/specs/C2_review.md (R1-R13) and
docs/specs/C2b_addendum.md (B1-B20).
"""
