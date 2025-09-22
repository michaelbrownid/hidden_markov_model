# train HMM on multiple inputs accumulating Viberbi counts
import argparse
import pathlib
import sys
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
        
        print("################################ datafile",datafile)
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
        nlpPerSymbol = result[0]/len(hmmAcc.targetOutputSeq)
        thisresult = ( hmmAcc.targetOutputSeq, nlpPerSymbol )
        print("thisresult",thisresult)
        resultNlpPerSymbol.append( thisresult )

        #### get Viterbi path and add counts
        viterbiPath = hmmAcc.backwardAlign("START")
        hmmAcc.addCountsFromViterbi( viterbiPath )


    #### estimate model from accumulated counts
    hmmAcc.estimateModelFromCounts()
    ofp = open(args.newmodel,"w")
    print(hmmAcc,file=ofp)
    ofp.close()

    #### output scores
    print("resultNlpPerSymbol")
    for xx in resultNlpPerSymbol:
        print(xx[0],xx[1])
        
    #### check against old
    hmmOld = HMM_VEC.HMM()
    hmmOld.readJSON("NEWMODEL_0.json")
    hmmAcc.diffmax(hmmOld)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model file name viz RB_model.json")
    parser.add_argument("--newmodel", help="newly estimated model file name to write viz RB_model_estimated.json")
    parser.add_argument("--dataListTSV", help="list of TSV input objects viz: RB_06.json.embeddings. Code converts to JSON using generateData and loads on the fly.")
    args = parser.parse_args()
    main(args)
