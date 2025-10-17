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
def findkv( text,key ):
    for line in text.split("\\n"):
        if key in line:
            (kk,vv) = line.split(":")
            return(vv)
    return(None)

################################
def computeStateUse( generation ):
    """ Compute number of times each output state is used in sequence:
    [
{"transName": "START", "transTo": "NW_1", "transCh": 1, "transNlp": 0.7058859755681709, "myprod": 0.7058859755681709},
{"outputName": "NW_1", "outputProbOut": "NW_1_out", "outputNlp": -2451.248952828494},
{"transName": "NW_1", "transTo": "NW_1", "transCh": 1, "transNlp": 0.4855071907818962, "myprod": -2450.0575596621443},
{"outputName": "NW_1", "outputProbOut": "NW_1_out", "outputNlp": -2435.748706798363},
{"transName": "NW_1", "transTo": "NW_4", "transCh": 4, "transNlp": 2.3966100989881265, "myprod": -4883.409656361519},
{"outputName": "NW_4", "outputProbOut": "NW_4_out", "outputNlp": -2353.251188221816},
{"transName": "NW_4", "transTo": "NW_2", "transCh": 2, "transNlp": 1.6072353398789114, "myprod": -7235.053609243457},
{"outputName": "NW_2", "outputProbOut": "NW_2_out", "outputNlp": -2530.1675721607935},
{"transName": "NW_2", "transTo": "NW_2", "transCh": 2, "transNlp": 0.4858354944624962, "myprod": -9764.735345909787},
{"outputName": "NW_2", "outputProbOut": "NW_2_out", "outputNlp": -2486.5432768963633},
{"transName": "NW_2", "transTo": "NW_1", "transCh": 1, "transNlp": 1.485275269813678, "myprod": -12249.793347536337},
{"outputName": "NW_1", "outputProbOut": "NW_1_out", "outputNlp": -2444.308004128949},
{"transName": "NW_1", "transTo": "NW_1", "transCh": 1, "transNlp": 0.4855071907818962, "myprod": -14693.615844474503},
{"outputName": "NW_1", "outputProbOut": "NW_1_out", "outputNlp": -2448.9816089414285},
{"transName": "NW_1", "transTo": "NW_1", "transCh": 1, "transNlp": 0.4855071907818962, "myprod": -17142.11194622515},
{"outputName": "NW_1", "outputProbOut": "NW_1_out", "outputNlp": -2478.031128141855},
{"transName": "NW_1", "transTo": "NW_0", "transCh": 0, "transNlp": 2.9457238042007456, "myprod": -19617.197350562805},
{"outputName": "NW_0", "outputProbOut": "NW_0_out", "outputNlp": -3080.952762923023},
{"transName": "NW_0", "transTo": "LAST", "transCh": 6, "transNlp": 0.7205471547485598, "myprod": -22697.429566331077}
]
    """

    result = {}
    for line in generation:
        if "outputProbOut" in line:
            result[line["outputProbOut"]] = result.get(line["outputProbOut"],0) + 1
    return(result)

################################
def main( args ):
    hmmAcc = HMM_VEC.HMM()
    hmmAcc.readJSON(args.model)

    ################################
    #### sample output from model with rejection
    done = False
    while not done:
        mygenerate = hmmAcc.generate()
        if len(mygenerate)<3: continue
        mylen = (len(mygenerate)-1)/2 # first trans and output+trans per output TODO: for now under model structure
        nlpPer = mygenerate[-1]["myprod"]/mylen
        print("generate attempt: mylen",mylen,"nlpPer",nlpPer, file=sys.stderr)
        if mylen<=args.maxlen and mylen>=args.minlen and nlpPer<=args.maxScore: done = True
        
    print("[\n%s\n]\n" % ",\n".join([ json.dumps(xx) for xx in mygenerate]))

    ################################    
    # - Compute negative log probability of each of the $S$ bridge shots
    #   (tsv vectors) under each of the $M$ model states NLP[M,S]

    #### read the objects file as data file
    basename = pathlib.Path(args.objects).name
    datjson = generateData.dataTSVToJSON( args.objects, basename)
    data = HMM_DAT.Data( datjson, isFile=False)
    dl = data.listDataSeq()
    hmmAcc.targetOutputSeq = dl[0] # or basename
    (hmmAcc.targetOutputNames, hmmAcc.targetOutput, hmmAcc.targetOutputName2Idx) = data.dataSeqMatrixFromName(hmmAcc.targetOutputSeq )

    hmmAcc.computeNlpModelStateByObjects()

    ################################
    # - Assign the best shot to each generated state by sorting the NLP and
    #   taking the top-k where $k$ is the number of times that state is used
    #   in the sequence (randomly assigned if multiple uses).

    #### tally number of times each state is used in mygenerate
    stateUse = computeStateUse( mygenerate )
    print("stateUse",stateUse)

    #### assign output states to output indices maintaining already used
    used = dict()
    selected = list()
    for line in mygenerate:
        if "outputProbOut" in line:
            state = line["outputProbOut"]
            (outputIdx, outputNlp) = hmmAcc.model[ hmmAcc.nameToIndex[ state ] ].selectNlpObject(used)
            print("*** selected",outputIdx, outputNlp,line)
            selected.append(outputIdx)
            used[outputIdx] = 1
        else:
            print("***",line)
    print("selected",selected)
    
    # - Use the jsv file to pick out begin,end timestamps and produce edit list.
    objectText = open(args.objectsText).read().splitlines()
    print("len(objectText)",len(objectText))
    print("********************************")
    for ss in selected:
        start = findkv(objectText[ss],"start_timestamp")
        end =   findkv(objectText[ss],"end_timestamp")
        print(ss,start,(float(end)-float(start)),objectText[ss])
        
    # - output those vectors in a new data point sequence, generatedData

    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model file name viz RB_model.json")
    parser.add_argument("--minlen", type=int, default=9, help="minimum number of shots")
    parser.add_argument("--maxlen", type=int, default=16, help="maximum number of shots")
    parser.add_argument("--maxScore", type=float, default= -2000, help="maximum score (negative log probability) per shot default: -2000")
    parser.add_argument("--objects", help="objects file. row of tsv vectors (embeddings)")
    parser.add_argument("--objectsText", help="text for each of the objects in same order")
    
    args = parser.parse_args()
    main(args)
