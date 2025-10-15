# train HMM on multiple inputs accumulating Viberbi counts
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
    
    #### cycle through training objects, parse, accumulate counts
    resultNlpPerSymbol = []
    for datafile in open(args.dataListTSV).read().splitlines():

        if datafile[0] == "#": continue
        
        print("################################ datafile",datafile, file=sys.stderr)
        #### clear memo computation
        hmmAcc.clearMemo()
        
        #### get data
        basename = pathlib.Path(datafile).name
        datjson = generateData.dataTSVToJSON( datafile, basename)
        data = HMM_DAT.Data( datjson, isFile=False)
        dl = data.listDataSeq()
        hmmAcc.targetOutputSeq = dl[0] # or basename
        (hmmAcc.targetOutputNames, hmmAcc.targetOutput, hmmAcc.targetOutputName2Idx) = data.dataSeqMatrixFromName(hmmAcc.targetOutputSeq )

        #### parse
        result = hmmAcc.backward("START")
        nlpPerSymbol = result.vitProb/len(hmmAcc.targetOutput)
        thisresult = ( hmmAcc.targetOutputSeq, nlpPerSymbol, len(hmmAcc.targetOutput))
        print("thisresult",thisresult, file=sys.stderr)
        resultNlpPerSymbol.append( thisresult )

        #### get Viterbi path and add counts
        viterbiPath = hmmAcc.backwardAlign("START")
        hmmAcc.addCountsFromViterbi( viterbiPath )


    #### compute average nlpPerSymbol
    nlpPerSymbolAvg = sum( [ xx[1] for xx in resultNlpPerSymbol ] ) / len(resultNlpPerSymbol)
                           
    #### estimate model from accumulated counts
    hmmAcc.estimateModelFromCounts()
    ofp = open(args.newmodel,"w")
    print(hmmAcc,file=ofp)
    ofp.close()

    #### output info
    info = dict()
    info["callArgs"]="HMM_VEC_TRAIN_ACC.py %s" % args
    info["nlpPerSymbolAvg"] = nlpPerSymbolAvg
    info["nlpPerSymbolNumObj"] = len(resultNlpPerSymbol)
    tsvrows = [ "\t".join( map(str, row) ) for row in resultNlpPerSymbol]
    tsvdata = "\n".join( tsvrows )
    info["resultNlpPerSymbol"]= tsvdata
    print(json.dumps(info, indent=2))
    
    # #### check against old
    # hmmOld = HMM_VEC.HMM()
    # hmmOld.readJSON("NEWMODEL_0.json")
    # hmmAcc.diffmax(hmmOld)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model file name viz RB_model.json")
    parser.add_argument("--newmodel", help="newly estimated model file name to write viz RB_model_estimated.json")
    parser.add_argument("--dataListTSV", help="list of TSV input objects viz: RB_06.json.embeddings. Code converts to JSON using generateData and loads on the fly.")
    args = parser.parse_args()
    main(args)
