# diff parameters of two HMM with same structure
import argparse
import pathlib
import sys
import json
try:
    from . import HMM_DAT
    from . import HMM_VEC
    from . import generateData
except ImportError:
    import HMM_DAT
    import HMM_VEC
    import generateData

################################
def main( args ):
    #### model to accumulate counts
    hmmAcc = HMM_VEC.HMM()
    hmmAcc.readJSON(args.model)
    hmmAcc.clearCounts()
    
    #### check against old
    hmmOld = HMM_VEC.HMM()
    hmmOld.readJSON(args.goldmodel)
    hmmAcc.diffmax(hmmOld)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model file name viz RB_model.json")
    parser.add_argument("--goldmodel", help="correct gold-standard model to compare to")
    args = parser.parse_args()
    main(args)
