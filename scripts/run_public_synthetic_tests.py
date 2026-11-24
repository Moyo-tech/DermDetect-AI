from pathlib import Path
import sys,unittest
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
suite=unittest.defaultTestLoader.discover(str(R/'tests'))
r=unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(not r.wasSuccessful())
