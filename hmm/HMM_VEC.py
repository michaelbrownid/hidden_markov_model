import numpy as np
import math
import argparse
import copy
import json
import sys
from . import HMM_DAT

################################
# Conventions
"""

- ranges are left closed, right open [begin, end)
- [0,1) is single position at 0
- [2,2) is nothing at position 2
- [2,1) is nothing at position 2

- inputs are probabilities but are stored as -log(prob)

- always start at START

- always end at LAST

- path probabilities in memo tables are stored as (
Viterbi probability,
Viterbi Consumed Length,
Viterbi best choice,
localNlpOut,
localNlpTrans,
summed probability all paths
)

"""

################################
# all probabilities are stored at -log(p)
# https://gasstationwithoutpumps.wordpress.com/2014/05/06/sum-of-probabilities-in-log-prob-space/
def nl( prob ):
    if prob==0.0:
        #zero probabilties are special, return infintiy=large number
        return(1.0E+300)
    else:
        return(-math.log(prob))

def probFromNlp(nl):
    return(math.exp(-nl))
           
def nladd( nl1, nl2):
    """ return -log( exp(-nl1) + exp(-nl2) ). 

    p1 = math.exp( - nl1 )
    p2 = math.exp( - nl2 )
    return( nl( p1+p2) )

    optimize by nl1 is smaller than nl2 or prob bigger. -log(p) large means low prob
    -log( exp(-nl1)*(1 + exp(-nl2 + nl1)))
    nl1 -log( 1 + exp(-nl2+nl1))
    """

    if nl1>nl2:
        (nl2,nl1)=(nl1,nl2)
    #print("nladd",nl1,nl2)
    return(nl1 + nl(1.0+math.exp(-nl2+nl1)))

################################
class HMMStateOutputMVN:
    def __init__(self, json):
        self.name = None
        self.mean = None
        self.sd = None
        self.readJSON( json )

    def nlp( self, output ):
        # R=-log(dnorm). the output nl probability of this state
        # outputting output this is currently a diagonal variance
        # Multi-Variate Normal prob = \product_i dnorm( output_i,
        # mean=self.param["mean"][i], sd=self.param["sd"][i])

        myprod = nl(1.0) # 0.0
        for ii in range(len(output)):
            myNorm = math.sqrt(2*math.pi*self.sd[ii]*self.sd[ii])
            Zscore = (output[ii]-self.mean[ii])/self.sd[ii]
            #prob = (1/myNorm)*math.exp(-0.5*Zscore*Zscore)
            #logprob= -math.log(myNorm) - (0.5*Zscore*Zscore)
            nlp = math.log(myNorm) + (0.5*Zscore*Zscore)
            myprod = myprod + nlp
        return(myprod)

    def generate( self ):
        #### generate diagonal MVG point and nlp
        point = np.random.normal( loc = self.mean, scale= self.sd)
        return( (point, self.nlp(point) ) )
                
    def __str__(self):
        tmp=[]
        tmp.append('"type": "HMMStateOutputMVN"')
        tmp.append('"name": "%s"' % self.name)
        tmp.append('"mean":  %s' % json.dumps(self.mean))
        tmp.append('"sd":  %s' % json.dumps(self.sd))
        return('{\n%s\n}\n' % ",\n".join(tmp))

    def readJSON(self, json):
        if json["type"] != "HMMStateOutputMVN": print("ERROR", json, "is not HMMStateOutputMVN")
        self.name = json["name"]
        self.mean = json["mean"]
        self.sd =   json["sd"]

################################
class HMMState:
    def __init__(self, json):
        self.name = None
        self.nexts = [] # next state
        self.nextp = [] # nl prob of next state
        self.prevs = [] # previous state that arrive here
        self.prevp = [] # nl prob previous state that arrive here
        self.prevComputed=False
        self.probOut = None
        self.readJSON( json )
        
    def __str__(self):
        tmp=[]
        tmp.append('"type": "HMMState"')
        tmp.append('"name": "%s"' % self.name)
        tmpnext = []
        for ii in range(len(self.nexts)):
            tmpnext.append('["%s", %f]' % (self.nexts[ii], probFromNlp( self.nextp[ii] ) ))
        tmp.append('"next": [ %s ]' % ",".join(tmpnext))
        tmp.append('"probOut": "%s"' % self.probOut)
        return('{\n%s\n}\n' % ",\n".join(tmp))


    def readJSON(self, json):
        if json["type"] != "HMMState": print("ERROR", json, "is not HMMState")
        self.name = json["name"]
        for (nexts, nextp) in json["next"]:
            self.nexts.append(nexts)
            self.nextp.append(nl(nextp))
        self.probOut = json["probOut"]

        
    def localLengths(self):
        # what lengths this state can derive
        if self.probOut == "SILENT":
            return([0])
        else:
            return([1])

################################        
class HMM:
    def __init__( self ):
        self.jsonfile = None
        self.json = None
        self.nameToIndex = {}
        self.typeToIndex = {}
        self.model = []
        self.memo = {}
        self.memoForward = {}
        self.targetOutput = None
        self.targetOutputNames = None
        self.band = 256 # abs(thisstate-begin) cannot be larger than self.band

    ################################        
    def name2state( self, name ):
        return(self.model[ self.nameToIndex[ name ] ] )

    ################################        
    def key(self, state,length):
        return("%s-%s" % (str(state),str(length)))

    def key(self, state,begin,end):
        return("%s-%s-%s" % (str(state),str(begin),str(end)))

    ################################
    def readJSON( self, jsonfile ):

        #### read in model
        self.jsonfile = jsonfile
        self.json = json.load( open(self.jsonfile) )

        #### map names and states
        for ii in range( len(self.json) ):
            this = self.json[ii]
            myname = this["name"]
            mytype = this["type"]

            if myname not in self.nameToIndex:
                self.nameToIndex[myname] = ii
            else:
                print(f"ERROR. myname {myname} is already taken")

            if mytype not in self.typeToIndex:
                self.typeToIndex[mytype] = [ii]
            else:
                self.typeToIndex[mytype].append(ii)

        #### fill out model states from JSON in order
        for ii in range( len(self.json) ):
            this = self.json[ii]
            myname = this["name"]
            mytype = this["type"]

            if mytype == "HMMStateOutputMVN":
                self.model.append( HMMStateOutputMVN( this ) )

            if mytype == "HMMState":
                self.model.append( HMMState( this ) )

    ################################
    def __str__(self):
        #### output string of list of json states based on the model states
        tmp=[]
        for ii in range( len(self.model) ):
            this = self.model[ii]
            tmp.append(str(this))
        return("[\n%s\n]\n" % ",\n".join(tmp))

    ################################
    def generate( self ):

        """Generate random strings from the HMM"""

        myprod = nl(1.0)
        here = "START" # by convention always start at START
        while here != "LAST": # by convention always end at LAST
            herest = self.name2state(here)
            # output if not SILENT and transistion
            if not herest.probOut == "SILENT":
                (output,nlp) = self.name2state(herest.probOut).generate()
                print(json.dumps(output.tolist()), "#output",herest.name,  herest.probOut, "output", "nlp", nlp)
                myprod = myprod + nlp

            myprob = [probFromNlp(xx) for xx in herest.nextp]
            ch = np.random.choice(len(myprob), p=myprob)
            myprod = myprod + herest.nextp[ch]
            print("# trans",herest.name,  "to", herest.nexts[ch], "ch", ch, "transNlp", herest.nextp[ch], "myprod", myprod)
            here = herest.nexts[ch]

    ################################
    def backward( self, thisstate, begin=None, end=None):

        """returns the (backward Vierbi probability, best choice,
           summed all paths prob) of this state deriving [begin:end)
           of self.targetOutput.  P(thisstate->[begin,end)), the most
           natural language interpretation (inside). End function
           parameter is usually fixed at the end of the string for
           regular grammars.

        """

        #### fill in boundaries on first call
        if begin is None: begin=0
        if end is None: end=len(self.targetOutputNames)

        if self.key(thisstate,begin,end) in self.memo:
            return self.memo[self.key(thisstate,begin,end)]
        
        # at LAST
        if thisstate=="LAST":
            if (end-begin)!=0:
                # LAST cannot derive anything, so prob=0.0
                result = (nl(0.0),0,"None",0,0,nl(0.0))
            else:
                result = (nl(1.0),0,"None",0,0,nl(1.0)) # prob=1.0
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        if (end-begin)<0:
            # impossible derivation
            result = (nl(0.0),0,"None",0,0,nl(0.0))
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        this = self.name2state(thisstate)

        print("exploring", this.name, begin, end)

        #### TODO: implement banding which requires constraints on states
            
        # start with probability 0
        overallresult = (nl(0.0),0,"None",0,0,nl(0.0))
        Zsum = nl(0.0)

        # output the data point at begin
        if this.probOut != "SILENT":
            if (end-begin)<=0:
                # impossible derivation for NOT SILENT
                result = (nl(0.0),0,"None",0,0,nl(0.0))
                self.memo[self.key(thisstate,begin,end)]=result
                return(result)
            localNlp = self.name2state(this.probOut).nlp( self.targetOutput[begin] )
            localLen = 1
        else:
            localNlp = nl(1.0) # silent output nothing with prob 1.0
            localLen = 0

        # cycle through next choices
        for ch in range(len(this.nexts)):

            # transistion
            transNlp = this.nextp[ch]

            # continue the derivation
            rhs = self.backward( this.nexts[ch], begin+localLen, end)

            ViterbiNlp = localNlp + transNlp + rhs[0]
            Zsum = nladd(Zsum, localNlp + transNlp + rhs[-1])

            if ViterbiNlp<overallresult[0]:
                # update viterbi, if nl is smaller, probability is larger
                overallresult = (ViterbiNlp, localLen, this.nexts[ch], localNlp, transNlp, nl(0.0))
            #print("***",this.name,begin,end,"ch",ch,"ViterbiNlp",ViterbiNlp, "localNlp",localNlp,"transNlp",transNlp,"Zsum",Zsum,"overallresult",overallresult)
            
        # put Zsum back in after summing over all
        overallresult = (overallresult[0],overallresult[1],overallresult[2],overallresult[3],overallresult[4],Zsum)
        self.memo[self.key(thisstate,begin,end)]=overallresult
        return(overallresult)

    ################################
    def backwardAlign( self, thisstate, begin=None, end=None):
        """Output the Viterbi path once backward has been computed"""
        #### fill in boundaries on first call
        if begin is None: begin=0
        if end is None: end=len(self.targetOutputNames)

        statepath = {}
        statepath["keys"]=["targetOutputSeq","thisstate","begin","end","localLen","dataNames","nextstate","localNlpOut","localNlpTrans","viterbiprob","sumprob"]
        statepath["values"] = []
        
        while thisstate!="LAST":
            (viterbiprob, localLen, viterbichoice, localNlpOut, localNlpTrans, sumprob) = self.memo[self.key(thisstate,begin,end)]
            statepath["values"].append( [self.targetOutputSeq, thisstate,begin,end, localLen, self.targetOutputNames[begin], viterbichoice, localNlpOut, localNlpTrans, viterbiprob, sumprob] )

            #### conusume localLen and move to next best state
            begin = begin+localLen
            thisstate = viterbichoice

        return(statepath)
    
    ################################
    def computePrev( self, thisstate ):

        """Inital call to the start state (thisstate=0), compute prevs and
        prevp, the previous states that arrive at thisstate for use in
        the forward probability computation
        """

        if self.state[thisstate].prevComputed:
            # already done
            return()

        # cycle through nexts
        for ii in range(len(self.state[thisstate].nexts)):
            nn = self.state[thisstate].nexts[ii]
            pp = self.state[thisstate].nextp[ii]
            self.state[nn].prevs.append(thisstate)
            self.state[nn].prevp.append(pp)

        self.state[thisstate].prevComputed=True

        for ii in range(len(self.state[thisstate].nexts)):
            nn = self.state[thisstate].nexts[ii]
            # print "computePrev", thisstate, ">>", nn
            self.computePrev( nn )


    ################################
    def forward( self, thisstate, begin, end):

        """returns the forward viterbi probability, best choice, summed prob
        of the start state deriving [begin, end) and arriving in
        thisstate. begin is usually constant at 0.

        """

        # print "forward exploring", thisstate, begin, end

        if self.key(thisstate,begin,end) in self.memoForward:
            return self.memoForward[self.key(thisstate,begin,end)]

        this = self.state[thisstate]

        # at begining (0 by convention)
        if thisstate == 0:
            if (end-begin)>0:
                result = (0.0,-1,0.0)
            else:
                result = (1.0,-1,1.0)
            self.memoForward[self.key(thisstate,begin,end)]=result
            return(result)

        if (end-begin)<0:
            result = (0.0,-1,0.0)
            self.memoForward[self.key(thisstate,begin,end)]=result
            return(result)

        # cycle through choices
        overallresult = (-1.0,-1,-1)
        Zsum = 0.0
        for ch in range(len(this.prevs)):

            # previous state forward psf, prev make possible emission pse, prev trans to thisstate pst, stop

            prev = self.state[this.prevs[ch]]

            if not prev.silent:
                if (end-begin)<1:
                    continue
                else:
                    pse = prev.out[self.alphaToInd[self.targetOutput[end-1]]]
                    prevend = end-1
            else:
                pse = 1.0
                prevend = end

            pst = this.prevp[ch]

            psf = self.forward( this.prevs[ch], begin, prevend)

            thisprob = pse*pst*psf[0]
            Zsum = Zsum + pse*pst*psf[2]
            if thisprob>overallresult[0]:
                overallresult = (thisprob, this.prevs[ch],-1)

        storeresult = (overallresult[0],overallresult[1],Zsum)
        #print "forward storing", this.name, begin, end, storeresult
        self.memoForward[self.key(thisstate,begin,end)]=storeresult
        return(storeresult)

################################
def main():
    myhmm = HMM()
    myhmm.readJSON(sys.argv[1])

    if False:
        print(str(myhmm))

    if False:
        print("================================")
        myhmm.generate()
        print("================================")

    if True:
        # get some data and run forward
        data = HMM_DAT.Data( sys.argv[2])
        print("data.listDataSeq()",data.listDataSeq())

        myhmm.targetOutputSeq = "RB_06"
        (myhmm.targetOutputNames, myhmm.targetOutput) = data.dataSeqMatrixFromName(myhmm.targetOutputSeq )

        result = myhmm.backward("START")
        print("result",result)

        viterbiPath = myhmm.backwardAlign("START")
        print("viterbiPath")
        print("\t".join(viterbiPath["keys"]))
        for vv in viterbiPath["values"]:
            print("\t".join([str(xx) for xx in vv]))
        
main()    
