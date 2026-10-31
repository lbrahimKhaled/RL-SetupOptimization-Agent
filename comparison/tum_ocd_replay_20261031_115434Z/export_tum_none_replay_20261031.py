from pathlib import Path
import importlib.util
import sys
root=Path('/Users/ibrahimkhaled/Documents/RL-SetupOptimization-Agent')
path=root/'scripts/comparison/export_tum_lap.py'
sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('export_tum_reference',path)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.VARIANTS=('none',)
mod.main()
