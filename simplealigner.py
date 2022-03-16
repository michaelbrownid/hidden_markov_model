import sys
import numpy as np
import math
from itertools import groupby

def rle(input_string):
    return [(len(list(g)), k) for k,g in groupby(input_string)]

def cigar( countOp ):
    #optocigar = {0:"refdel", 1:"refins", 2:"miss", 3:"match"}
    optocigar = {0:"D", 1:"I", 2:"X", 3:"="}
    totalerrors = 0
    totalref = 0
    totalread = 0
    out = []
    for (count,op) in countOp:
        out.append("%d%s" % (count,optocigar[op]))
        if op!=3: totalerrors+=count
        if op!=1: totalref+=count
        if op!=0: totalread+=count
    return({"cigarstr":"".join(out), "totalerrors":totalerrors,"totalref":totalref, "totalread":totalread})
        
def backtrack( track, ii, jj ):
    # [[[[[[[[[[[[[[None, 1], 1], 1], 1], 1], 1], 1], 1], 1], 1], 2], 2], 3], 3])
    if (ii+jj)==0: return(None)
    this = track[ii,jj]
    if this==0:
        return([backtrack(track, ii-1, jj),0])
    if this==1:
        return([backtrack(track, ii, jj-1),1])
    if this==2:
        return([backtrack(track, ii-1, jj-1),2])
    if this==3:
        return([backtrack(track, ii-1, jj-1),3])

def backtrackbackwards( track, ii, jj, maxii,maxjj ):
    if (ii==maxii) and (jj==maxjj): return(None)
    this = track[ii,jj]
    #print("backtrackbackwards",ii,jj,this,maxii,maxjj)
    if this==0:
        return([0,backtrackbackwards(track, ii+1, jj,maxii,maxjj)])
    if this==1:
        return([1,backtrackbackwards(track, ii, jj+1,maxii,maxjj)])
    if this==2:
        return([2,backtrackbackwards(track, ii+1, jj+1,maxii,maxjj)])
    if this==3:
        return([3,backtrackbackwards(track, ii+1, jj+1,maxii,maxjj)])

def collapse( xx, result ):
    if not isinstance(xx,list):
        if xx is not None:
            result.append(xx)
    else:
        collapse(xx[0], result)
        collapse(xx[1], result)

def findmin( cc ):
    ss = sorted( zip(cc,range(len(cc))), key = lambda xx: xx[0])
    return(ss[0])

def neglogsumneglogprobs( lp ):
    # TODO: could do pairwise
    mysum = sum( [math.exp(-xx) for xx in lp] )
    return(-math.log(mysum))
    #return(sum(lp))

def printmatrix( ref, read, dat):
    myref="_"+ref+"_"
    myread="_"+read+"_"
    print("ref_v\t"+"\t".join([char for char in myread]))
    for row in range(len(myref)):
        print(myref[row]+"\t"+"\t".join(["%.3f" % (xx) for xx in dat[row,:]]))
        
################################
def alignforward(p, s1, s2, doSum=False):
    # ref = s1. read = s2. so insert / deletes can be interpreted.

    ls1=len(s1)
    ls2=len(s2)

    choice = [0.0]*4

    # set up cost matrix costfwd[ref][read]

    # fill in table (42 is NA)
    # this is the forward table: costfwd[J,t] = P(S->[1,t] and in state J)
    costfwd = 42*np.ones( (ls1+2,ls2+2), dtype=np.float32)
    track = 42*np.ones( (ls1+2,ls2+2), dtype=np.int32)

    """
    0,0=0
    1,0=del 1st state
    i,0=del ith state ...
    0,1=ins. both match and delete move to state!
    1,1=min(ins,match,delete)
    """

    costfwd[0,0] = 0.0 # align nothing to nothing has 0 cost, can do nothing else

    # basis all delete the ref, align to nothing read[0]. 
    for refi in range(1,ls1+1):
        costfwd[refi,0] = costfwd[refi-1,0] + p["tdel"]
        track[refi,0] = 0 # delete ref

    # basis 0th state outputs symbols as insertions staying in 0th state
    for readi in range(1,ls2+1):
        costfwd[0,readi] = costfwd[0,readi-1] + (p["tins"]   + p["oins"])

    for refi in range(1,ls1+1):
        for readi in range(1,ls2+1):
            """ f[ref,read]
            f[-1, 0] = delete ref moves but read does not
            f[ 0,-1] = insert ref does not move but read does
            f[-1,-1] = match both move
            """
            choice[0] = (p["tdel"]   + 0.0)       + costfwd[refi-1,readi]
            choice[1] = (p["tins"]   + p["oins"]) + costfwd[refi,readi-1]

            if ( s1[refi-1] != s2[readi-1]):
                choice[2]  = (p["tmatch"] + p["omiss"]) + costfwd[refi-1,readi-1] # mismatch
                choice[3]  = 99.9E+99
            else:
                choice[2]  = 99.9E+99
                choice[3]  = (p["tmatch"] + p["omatch"]) + costfwd[refi-1,readi-1] # match

            if not doSum:
                (mymin, mychoice) = findmin(choice)
            else:
                mymin = neglogsumneglogprobs( choice )
                mychoice = 42

            costfwd[refi,readi] = mymin
            track[refi,readi] = mychoice

    # set the last row  and col
    for readi in range(ls2+2):
        costfwd[ls1+1,readi] = 0.0
    for refi in range(ls1+2):
        costfwd[refi,ls2+1] = 0.0

    # final cost at end of read and model
    finalcost= costfwd[ls1,ls2]

    if not doSum:
        finalpath=[]
        collapse(backtrack( track, ls1, ls2 ), finalpath)
        final = cigar(rle(finalpath))
    else:
        final={}
    final["finalcost"]=finalcost
    final["cost"]=costfwd
    return( final )

################################
def alignbackward (p, s1, s2, doSum=False):
    # ref = s1. read = s2. so insert / deletes can be interpreted.

    ls1=len(s1)
    ls2=len(s2)

    choice = [0.0]*4

    costback = 42*np.ones( (ls1+2,ls2+2), dtype=np.float32)
    track = 42*np.ones( (ls1+2,ls2+2), dtype=np.int32)

    # states derive nothing at end
    costback[ls1+1,ls2+1] = 0.0
    for refi in range(ls1,-1,-1):
        costback[refi,ls2+1] = costback[refi+1,ls2+1] + p["tdel"]
        track[refi,ls2+1] = 0 # delete ref. to step along ref to get to last state

    # the last state can only derive insertions
    for readi in range(ls2,-1,-1):
        costback[ls1+1,readi] = costback[ls1+1,readi+1] + p["tins"] + p["oins"]
        track[ls1+1,readi] = 1 # delete read = insert to ref

    for refi in range(ls1,0,-1):
        for readi in range(ls2,0,-1):

            choice[0] = (p["tdel"]   + 0.0)       + costback[refi+1,readi] # delete ref
            choice[1] = (p["tins"]   + p["oins"]) + costback[refi,readi+1] # delete read = insert to ref

            if ( s1[refi-1] != s2[readi-1]):
                choice[2]  = (p["tmatch"] + p["omiss"]) + costback[refi+1,readi+1] # mismatch
                choice[3]  = 99.9E+99
            else:
                choice[2]  = 99.9E+99
                choice[3]  = (p["tmatch"] + p["omatch"]) + costback[refi+1,readi+1] # match

            if not doSum:
                (mymin, mychoice) = findmin(choice)
            else:
                mymin = neglogsumneglogprobs( choice )
                mychoice = 42

            costback[refi,readi] = mymin
            track[refi,readi] = mychoice

    # set the first row and column to 0, only spacer for forward
    for refi in range(ls1+2):
        costback[refi,0] = 0
    for readi in range(ls2+2):
        costback[0,readi] = 0

    finalcost= costback[1,1]

    if not doSum:
        finalpath=[]
        collapse(backtrackbackwards( track, 1, 1, ls1+1, ls2+1 ), finalpath)
        final = cigar(rle(finalpath))
    else:
        final={}
    final["finalcost"]=finalcost
    final["cost"]=costback
    return(final)

################################

if __name__=="__main__":

    p={}
    p["tins"] = -math.log(0.05)
    p["tnotins"] = -math.log(1-0.05)
    p["tdel"] = -math.log(0.05)
    p["tmatch"] = -math.log(1.0-0.05-0.05)
    p["oins"]  = -math.log(0.25)
    p["omatch"] = -math.log(0.99)
    p["omiss"] = -math.log(0.01)

    # p={}
    # p["tins"] = 1
    # p["oins"]  = 1
    # p["tdel"] = 3
    # p["tmatch"] = 0
    # p["omatch"] = 0 
    # p["omiss"] =  1

    print("p",p)

    if True:
        res = alignforward(p,sys.argv[1], sys.argv[2], doSum=True)
        res2 = alignbackward(p,sys.argv[1], sys.argv[2], doSum=True)
        print(res["finalcost"])
        print(res2["finalcost"])

        print("====")
        printmatrix(sys.argv[1],sys.argv[2],res["cost"])
        print("====")
        printmatrix(sys.argv[1],sys.argv[2],res2["cost"])

        # print(res["totalerrors"],res["totalref"],res["finalcost"],res["cigarstr"])
        # print(res2["totalerrors"],res2["totalref"],res2["finalcost"],res2["cigarstr"])

        (nrow,ncol)=res["cost"].shape
        tt = np.exp(- (res["cost"][0:nrow-1, 0:ncol-1] +res2["cost"][1:nrow, 1:ncol]))
        print("====")
        print(tt)
        
        print(np.sum(tt,axis=0))
        print(np.sum(tt,axis=1))
        sys.exit(0)


    """
    res = align("CCCCACGTACGTACGT","ACGTACGTACGTTTTT")
    Alignments can be surprising:

    res (12.775767, {'cigarstr': '1X1=2X8=3X1=', 'totalerrors': 6, 'totalread': 16, 'totalref': 16})
    CCCCACGTACGTACGT
    x xx        xxx  = (0.9*0.01)^6 * (0.9*0.99)^10 = -log= 12.775767903
    ACGTACGTACGTTTTT

    CCCCACGTACGTACGT
    xxxx            xxxx = (0.05*0.25)^8 * (0.9*0.99)^12 = -log = 15.8 worse prob
        ACGTACGTACGTTTTT

    print("res",res)
    """

    # align two fasta files where order of sequences is the same
    refs = open(sys.argv[1]).read().splitlines()
    reads = open(sys.argv[2]).read().splitlines()

    allerrors =0
    allbases =0
    cc = 0
    while cc<len(refs):
        refid=refs[cc]
        readid=reads[cc]
        assert(refid==readid)
        cc+=1

        myref = refs[cc]
        myread = reads[cc]
        cc+=1

        res = align(myref,myread)
        myerr = float(res["totalerrors"])/float(res["totalref"])

        allerrors+=res["totalerrors"]
        allbases+=res["totalref"]

        print("%s\t%s\t%f\t%d\t%d" % (refid,readid,myerr,res["totalerrors"],res["totalref"]))

print("# overallerror = %f = %d / %d" % ( float(allerrors)/allbases, allerrors, allbases))
