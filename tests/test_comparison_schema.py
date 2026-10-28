"""comparison/: CSV headers match the schema in scripts/comparison/common.py, every CSV has its
.meta.json, and comparison/README.md holds the schema's column tables. Reads headers only (fast)."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('comparison_common', ROOT / 'scripts/comparison/common.py')
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)


@pytest.mark.parametrize('source', sorted(common.SCHEMAS))
def test_csv_headers_match_schema(source):
    files = sorted((ROOT / 'comparison' / source).glob('*.csv'))
    assert files, f'no CSV in comparison/{source}'
    for f in files:
        with f.open() as fh:
            assert fh.readline().rstrip('\n').split(',') == common.names(source), f.name
        meta = json.loads(f.with_suffix('.meta.json').read_text())
        assert meta['file'] == f.name and meta['source'] == source


def test_readme_tables_match_schema():
    text = (ROOT / 'comparison/README.md').read_text()
    for source in common.SCHEMAS:
        assert common.markdown_table(source) in text, source


def test_write_csv_rejects_unknown_columns(tmp_path):
    with pytest.raises(KeyError):
        common.write_csv(tmp_path / 'x.csv', 'tum', [{'not_a_column': 1.}])
    p = common.write_csv(tmp_path / 'y.csv', 'tum', [{'i': 3, 's_m': 1.23456789, 'drs': True, 'mu': float('nan')}])
    header, row = p.read_text().splitlines()
    vals = dict(zip(header.split(','), row.split(',')))
    assert vals['i'] == '3' and vals['s_m'] == '1.234568' and vals['drs'] == '1' and vals['mu'] == '' and vals['x_m'] == ''
