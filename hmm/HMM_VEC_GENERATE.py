# generate output from hmm
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

    #### sample output from model with rejection
    done = False
    while not done:
        myout = hmmAcc.generate()
        if len(myout)<3: continue
        mylen = (len(myout)-1)/2 # first trans and output+trans per output TODO: for now under model structure
        nlpPer = myout[-1]["myprod"]/mylen
        print("generate attempt: mylen",mylen,"nlpPer",nlpPer, file=sys.stderr)
        if mylen<=args.maxlen and mylen>=args.minlen and nlpPer<=args.maxScore: done = True
        
    print("[\n%s\n]\n" % ",\n".join([ json.dumps(xx) for xx in myout]))
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model file name viz RB_model.json")
    parser.add_argument("--minlen", type=int, default=9, help="minimum number of shots")
    parser.add_argument("--maxlen", type=int, default=16, help="maximum number of shots")
    parser.add_argument("--maxScore", type=float, default= -2000, help="maximum score (negative log probability) per shot default: -2000")
    args = parser.parse_args()
    main(args)
