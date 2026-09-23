"""Compatibility entry point for the current official-sampling snapshot."""
import argparse
from pathlib import Path
from build_official500 import ROOT, build

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT.parent/'phyRSI/robodawn_robotwin_harness_v72/results')
    p.add_argument('--output',type=Path,default=ROOT/'site')
    a=p.parse_args();build(a.results.resolve(),a.output.resolve())
