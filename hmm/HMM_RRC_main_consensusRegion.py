from . import HMM_RRC
from . import rrchmm_load_stitch
import rawcorr.bamClip as bc
import argparse
import numpy as np
import math
import sys

################################
def mutateX1( seq ):
    # apply all single base mutations to seq
    bases = "ACGT"
    all = { seq }
    
    # substitutions
    for ii in range(len(seq)):
        pre = seq[0:ii]
        post = seq[(ii+1):(len(seq))]
        for bb in range(len(bases)):
            toadd = pre+bases[bb]+post
            #print("sub",toadd)
            all.add(toadd)
            
    # deletions
    for ii in range(len(seq)):
        pre = seq[0:ii]
        post = seq[(ii+1):(len(seq))]
        toadd = pre+post
        #print("del",toadd)
        all.add(toadd)

    # insertions
    for ii in range(len(seq)):
        pre = seq[0:ii]
        post = seq[ii:(len(seq))]
        for bb in range(len(bases)):
            toadd=pre+bases[bb]+post
            #print("ins",toadd)
            all.add(toadd)

    return(all)

################################

def main(args):

    ################################
    # get the prediction data
    fullreads = rrchmm_load_stitch.loadStitch()
    fullreads.computeStitch( args.predictionsh5, args.featuresh5)


    ################################
    # set up a model for a region in each subread

    rcmodels = []
    rcmodelsSR = []
    fwmodels = []
    fwmodelsSR = []
    
    # m64002_220618_180054/140970246/248122_271717 21601 21618 RC READREGION GTGACAGCTGAATATAC GTGACGCTGAATGATAC REFRANGE 2431299 2431315
    for line in open(args.consensusRegion).read().splitlines():
        ff = line.split()
        (readid, begin, end ) = (ff[0],ff[1],ff[2])
        if args.propRef == "NA" and ff[3]=="FW":
            args.propRef = ff[5] # true FW ref for now

        readid=readid.encode() # binary storage
        begin = int(begin)
        end = int(end) # +1 # open right

        #### get predictions for full read
        mypreds = fullreads.getReadPredictions(readid)[begin:end]

        #### get original
        mybases = fullreads.getReadBaseFeatures(readid)[begin:end]

        #### decode for sanity
        predmax = np.argmax(mypreds,1)
        basemax = np.argmax(mybases,1)+20 # +20 to line up with 25 label_symbols
        for (bb,pp) in zip(basemax,predmax):
            if (bb!=pp):
                print("maxdisagree",readid,fullreads.label_symbols[bb],fullreads.label_symbols[pp],bb,pp)

        #### optionally remove max bursts
        if args.burstMaxRemove=="True":
            print("burstMaxRemove removing",np.sum(predmax==25))
            print("before mypreds.shape",mypreds.shape)
            mypreds = mypreds[predmax!=25,:]
            print("after mypreds.shape",mypreds.shape)
            reflen = int(ff[-1]) - int(ff[-2])
            ratio = (mypreds.shape[0]/ reflen)
            if ratio > 2.0 or ratio<0.5:
                # reject if difference over prop reflen is too big or small
                print("rejecting burst cleaned subread. ratio:",ratio,"reflen:",reflen,"mypreds.shape[0]",mypreds.shape[0])
                continue
            
        #### set up HMM for that full read
        myhmm = HMM_RRC.HMM( mypreds, readid )

        # not needed as I'm already taking the [begin:end] subregion
        # set up last state for this region
        #myhmm.states[end].name="last"

        if ff[3]=="FW":
            fwmodels.append(myhmm)
            fwmodelsSR.append(readid)
        else:
            rcmodels.append(myhmm)
            rcmodelsSR.append(readid)

    if args.dumpModels=="True":
        for myhmm in fwmodels:
            print("FW")
            print(myhmm)
        for myhmm in rcmodels:
            print("RC")
            print(myhmm)


    if args.productModels=="True":
        prodFW = HMM_RRC.HMM.productModels( fwmodels )
        prodRC = HMM_RRC.HMM.productModels( rcmodels )
        print("prodFW")
        print(prodFW)
        print("prodRC")
        print(prodRC)
        fwmodels = [prodFW]
        rcmodels = [prodRC]
        

    ################################
    # compute probs of truth

    def intersectSubreads(rctruth, fwtruth):
        #print("****************************************************************")
        allrc = []
        rcsum = 0.0
        rcsumV = 0.0
        cc = 0
        for myhmm in rcmodels:
            #print("================================")
            myhmm.targetOutput=rctruth
            myhmm.memo={}
            parse=myhmm.backward(0,0,len(myhmm.targetOutput))
            rcsum+=parse[2]
            rcsumV+=parse[0]
            #print("result RC ", parse[2], rcdat[cc])
            allrc.append( myhmm.backwardAlign(0,0,len(myhmm.targetOutput)) )
            cc += 1

        allfw = []
        fwsum=0.0
        fwsumV=0.0
        cc = 0
        for myhmm in fwmodels:
            #print("================================")
            myhmm.targetOutput=fwtruth
            myhmm.memo={}
            parse=myhmm.backward(0,0,len(myhmm.targetOutput))
            fwsum+=parse[2]
            fwsumV+=parse[0]
            #print("result FW ", parse[2], fwdat[cc])
            allfw.append( myhmm.backwardAlign(0,0,len(myhmm.targetOutput)) )
            cc += 1

        return({"allfw": allfw, "allrc": allrc, "rcsum":rcsum,"fwsum":fwsum, "sum": rcsum+fwsum, "rcsumV":rcsumV,"fwsumV":fwsumV, "sumV": rcsumV+fwsumV})

    if args.mutateX1=="True":
        ################################
        print("args.propRef",args.propRef)
        mutations = sorted(mutateX1(args.propRef))
        print("len(mutations)",len(mutations))
        allresults = []
        cc=0
        for myfw in mutations:
            resultLine = "%d %s " % (cc,myfw)
            myrc = bc.rc(myfw)
            results = intersectSubreads(myrc, myfw)
            resultLine += "sum %f fwsum %f rcsum %f " % (results["sum"],results["fwsum"],results["rcsum"])
            resultLine += "sumV %f fwsumV %f rcsumV %f" % (results["sumV"],results["fwsumV"],results["rcsumV"])
            # want smallest / most negative nullSavings  # TODO: length of subreads??
            reflen=len(myfw)
            numF=len(results["allfw"])
            numR=len(results["allrc"])
            log4 = math.log(4.0)

            # simple sum
            #nullSavingsSumV =  results["sumV"]
            #nullSavingsSum =   results["sum"]
            
            # sum / reflen
            #nullSavingsSumV =  results["sumV"]/reflen
            #nullSavingsSum =   results["sum"]/reflen

            # total null savings
            nullSavingsSumV =  results["sumV"]-(reflen * (numF+numR) * log4)
            nullSavingsSum =   results["sum"] -(reflen * (numF+numR) * log4)

            # per base:
            #nullSavingsSumV =  ( results["sumV"] / (reflen * (numF+numR)) ) - log4
            #nullSavingsSum  =  ( results["sum"]  / (reflen * (numF+numR)) ) - log4

            # per position
            #nullSavingsSumV =  (results["sumV"]-(reflen * (numF+numR) * log4))/reflen
            #nullSavingsSum =   (results["sum"] -(reflen * (numF+numR) * log4))/reflen
            
            # "incorrect" multiply
            #log4 = math.log(4.0)/math.log(10.0)
            #nullSavingsSumV =  (results["sumV"]-(reflen * (numF*numR) * log4))
            #nullSavingsSum =   (results["sum"] -(reflen * (numF*numR) * log4))

            # rcSumV null savings
            #nullSavingsSumV =  results["rcsumV"]-(reflen * (numR) * log4)
            #nullSavingsSum =   results["rcsum"] -(reflen * (numR) * log4)

            allresults.append( (resultLine, myfw, nullSavingsSumV, nullSavingsSum, reflen, numF, numR) )
            cc += 1
        ss = sorted(allresults,key=lambda xx: xx[2])
        firstScore=ss[0][2]
        secondScore=ss[1][2]
        firstSeq = ss[0][1]
        secondSeq = ss[1][1]
        if args.propRef == firstSeq:
            final = "correct"
        else:
            final = "wrong"
        print("RESULT:", final, args.propRef, firstScore, firstSeq, secondScore, secondSeq)

        for ssl in  ss:
            print("\t".join([str(xx) for xx in ssl]))

    if not args.mutateX1=="True":
        ################################
        if args.propRef=="null":
            myfw=""
            myrc=""
        else:
            myfw  = args.propRef
            myrc = bc.rc(myfw)
        result = intersectSubreads(myrc, myfw)

        numF=len(result["allfw"])
        numR=len(result["allrc"])

        print("readid total all", numF+numR, "sumprob:",result["sum"], "viterbiprob:",result["sumV"],)
        srXbaseCost = []

        for srii in range(len(result["allfw"])):
            srdat = result["allfw"][srii]
            print("readid", fwmodelsSR[srii].decode(), "allfw",srii, "sumprob",srdat["values"][0][-1],"viterbiprob",srdat["values"][0][-2])
            print("cigar",srdat["cigar"])
            carrycost=0.0
            thisbasecost = []
            for srbaseii in range(len(srdat["values"])):
                srbasedat = srdat["values"][srbaseii]
                #srdat["names"]: thisstate begin end name outputprobPerEvent outputprob myoutput oplist viterbiprob sumprob
                refbegin = srbasedat[1]
                outputprobPerEvent = srbasedat[4]
                myoutput = srbasedat[6]
                numoutput = len(myoutput)
                #print("# srii",srii,"refbegin",refbegin,"outputprobPerEvent",outputprobPerEvent,"numoutput",numoutput)
                if numoutput==0:
                    # if sr deletion then assign cost to next base
                    carrycost += outputprobPerEvent
                else:
                    if carrycost>0.0:
                        isi = "D"
                    else:
                        isi = ""
                    print("fw srii",srii,"refbase",refbegin,"outputprobPerEvent",outputprobPerEvent+carrycost,isi+"M",myoutput,"$")
                    thisbasecost.append(outputprobPerEvent+carrycost)

                    # # GOAL: compare difference in viteriprob vs thisbasecost
                    # # RESULT: the difference in vitrerbiProbs is exactly thisbasecost to 1%. CORRECT!
                    # nextbase = min(srbaseii+1, len(srdat["values"])-1)
                    # # prev deletion. Insertions are divided equally
                    # if carrycost>0.0:
                    #     thisbase=srbaseii-1
                    # else:
                    #     thisbase=srbaseii
                    # aa=outputprobPerEvent+carrycost
                    # bb=srdat["values"][thisbase][-2]-srdat["values"][nextbase][-2]
                    # print("thisbasecost", aa,bb,abs(aa-bb)<0.01)
                    
                    carrycost=0.0
                    for ii in range(1,numoutput):
                        print("fw srii",srii,"refbase",refbegin+ii,"outputprobPerEvent",outputprobPerEvent,"I", myoutput, "$")
                        thisbasecost.append(outputprobPerEvent)

            srXbaseCost.append(thisbasecost)

        for srii in range(len(result["allrc"])):
            srdat = result["allrc"][srii]
            print("readid", rcmodelsSR[srii].decode(), "allrc",srii, "sumprob",srdat["values"][0][-1],"viterbiprob",srdat["values"][0][-2])
            print("cigar",srdat["cigar"])
            carrycost=0.0
            thisbasecost = []
            for srbaseii in range(len(srdat["values"])):
                srbasedat = srdat["values"][srbaseii]
                #srdat["names"]: thisstate begin end name outputprobPerEvent outputprob myoutput oplist viterbiprob sumprob
                refbegin = srbasedat[1]
                outputprobPerEvent = srbasedat[4]
                myoutput = srbasedat[6]
                numoutput = len(myoutput)
                #print("# srii",srii,"refbegin",refbegin,"outputprobPerEvent",outputprobPerEvent,"numoutput",numoutput)
                if numoutput==0:
                    # if sr deletion then assign cost to next base
                    carrycost += outputprobPerEvent
                else:
                    if carrycost>0.0:
                        isi = "D"
                    else:
                        isi = ""
                    print("rc srii",srii,"refbase",len(myfw)-refbegin-1,"outputprobPerEvent",outputprobPerEvent+carrycost,isi+"M",myoutput,"$")
                    thisbasecost.append(outputprobPerEvent+carrycost)
                    carrycost=0.0
                    for ii in range(1,numoutput):
                        print("rc srii",srii,"refbase",len(myfw)-(refbegin+ii)-1,"outputprobPerEvent",outputprobPerEvent,"I", myoutput, "$")
                        thisbasecost.append(outputprobPerEvent)

            thisbasecost.reverse() # RC reverses the ref
            srXbaseCost.append(thisbasecost)

        for refbase in range(len(srXbaseCost[0])):
            prodP = 1.0
            prodQ = 1.0
            sumcost = 0.0
            for srii in range(len(srXbaseCost)):
                sumcost += srXbaseCost[srii][refbase]
                
                pi = math.exp(-srXbaseCost[srii][refbase])
                qi = 1.0-pi
                prodP *= pi
                prodQ *= qi
                # NOTE: 20 subreads should not overflow ... much

            print("refbase",refbase,"nlProdProbErr",-math.log(prodQ/(prodQ+prodP)), "sumcost", sumcost)
            
################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--featuresh5")
    parser.add_argument("--predictionsh5")
    parser.add_argument("--consensusRegion")
    parser.add_argument("--propRef", help="NA for --mutateX1 null for nothing")
    parser.add_argument("--burstMaxRemove", default="True")
    parser.add_argument("--mutateX1", default="True")
    parser.add_argument("--dumpModels", default="False")
    parser.add_argument("--productModels", default="False")
    args=parser.parse_args()
    main(args)

