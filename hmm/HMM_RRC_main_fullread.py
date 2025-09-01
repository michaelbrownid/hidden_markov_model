from . import HMM_RRC
from . import rrchmm_load_stitch
import argparse
import numpy as np

################################

def main(args):

    ################################
    # get the prediction data
    fullreads = rrchmm_load_stitch.loadStitch()
    fullreads.computeStitch( args.predictionsh5, args.featuresh5)

    #### get a readid
    readid = args.readId.encode()

    #### get predictions for full read
    mypreds = fullreads.getReadPredictions(readid)

    #### get original
    mybases = fullreads.getReadBaseFeatures(readid)

    #### decode for sanity
    predmax = np.argmax(mypreds,1)
    basemax = np.argmax(mybases,1)+20 # +20 to line up with 25 label_symbols

    for (bb,pp) in zip(basemax,predmax):
        if (bb!=pp):
            print("maxdisagree",fullreads.label_symbols[bb],fullreads.label_symbols[pp],bb,pp)
    
    #### set up HMM for that full read
    myhmm = HMM_RRC.HMM( mypreds )
    if args.truth == "base":
        myhmm.targetOutput = "".join( [ fullreads.label_symbols[xx].upper() for xx in basemax])
    elif args.truth == "pred":
        myhmm.targetOutput = "".join( [ fullreads.label_symbols[xx].upper() for xx in predmax])
    else:
        myhmm.targetOutput = open(args.truth).read().strip()

    if args.truthRC!="False":
        # reverse complement the reference
        rclut = {"A":"T", "C":"G", "G":"C", "T":"A", "a":"T", "c":"G", "g":"C", "t":"A"}
        rc = "".join( [rclut[xx] for xx in reversed(myhmm.targetOutput)] )
        myhmm.targetOutput = rc
        
    print(">>>>>")
    print("len(myhmm.targetOutput)",len(myhmm.targetOutput))
    print("len(myhmm.state)",len(myhmm.state))
    
    #print(myhmm.backward(0,0,len(myhmm.targetOutput)))
    #RecursionError: maximum recursion depth exceeded while getting the str of an object

    def targetBase(t): return(int(len(myhmm.targetOutput)*t+0.5))
    def hmmBase(t): return(int(len(myhmm.state)*t+0.5))
    for t in range(99,0,-1):
        tb = targetBase(t/100.0)
        hb = hmmBase(t/100.0)
        print(hb,tb,myhmm.backward(hb,tb,len(myhmm.targetOutput)), len(myhmm.memo))
        

    print("FULL",myhmm.backward(0,0,len(myhmm.targetOutput)))
    mycigar = myhmm.backwardAlign(0,0,len(myhmm.targetOutput))
    print("mycigar",mycigar)

################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--featuresh5")
    parser.add_argument("--predictionsh5")
    parser.add_argument("--truth")
    parser.add_argument("--truthRC",default="False")
    parser.add_argument("--readId")
    args=parser.parse_args()
    main(args)

