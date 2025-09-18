import numpy as np
import math
import argparse
import copy
import json
import sys
try:
    from . import HMM_DAT
except ImportError:
    import HMM_DAT

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
        self.countsData = [] # list of datapoints emitted by this state
        self.countsPosterior  = [] # corresponding emission posteriors. 1.0 for viterbi
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
        #### silence large parameters for now
        #tmp.append('"mean":  %s' % json.dumps(self.mean))
        #tmp.append('"sd":  %s' % json.dumps(self.sd))
        tmp.append('"countsData": "%s"' % self.countsData)
        tmp.append('"countsPosterior": "%s"' % self.countsPosterior)
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
        self.nextName2Idx = {} # state name to index in next
        self.prevs = [] # previous state that arrive here
        self.prevp = [] # nl prob previous state that arrive here
        self.prevName2Idx = {} # state name to index in prev
        self.prevComputed=False
        self.probOut = None

        # counts 2d [numPoints, len(self.nexts)]. For every point,
        # where it transistioned. Might be delta function for Viterbi
        # [0,1,0] or distribution for posterior [0.2, 0.7, 0.1]
        self.counts = [] 

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
        tmp.append('"counts": "%s"' % self.counts)
        #### sum counts
        npc = np.array(self.counts)
        tmp.append('"sumCounts": "%s"' % np.sum(npc,axis=0))
        return('{\n%s\n}\n' % ",\n".join(tmp))


    def readJSON(self, json):
        if json["type"] != "HMMState": print("ERROR", json, "is not HMMState")
        self.name = json["name"]
        for (nexts, nextp) in json["next"]:
            self.nexts.append(nexts)
            self.nextp.append(nl(nextp))
            self.nextName2Idx[nexts] = len(self.nexts)-1 # index
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

        # print("exploring", this.name, begin, end)

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
        """Output the Viterbi path once backward has been computed
        targetOutputSeq	thisstate	begin	end	localLen	dataNames	nextstate	localNlpOut	localNlpTrans	viterbiprob	sumprob
        RB_06	START	0	98	0	RB_06_S_0	RBS_4	-0.0	1.9427172517811961	-285430.5573645353	-285430.5573645353
        RB_06	RBS_4	0	98	1	RB_06_S_0	RBS_4	-3029.747402596579	1.96574149676392	-285432.50008178706	-285432.50008178706
        ...
        RB_06	RBS_0	95	98	1	RB_06_S_95	RBS_0	-3033.725616082737	1.9475645167672397	-8724.258364931258	-8724.258364931258
        RB_06	RBS_0	96	98	1	RB_06_S_96	RBS_1	-2787.045761956062	1.9363945661566904	-5692.480313365288	-5692.480313365288
        RB_06	RBS_1	97	98	1	RB_06_S_97	LAST	-2909.3023261969533	1.9313802215709954	-2907.3709459753823	-2907.3709459753823
        """
        
        #### fill in boundaries on first call
        if begin is None: begin=0
        if end is None: end=len(self.targetOutputNames)

        statepath = {}
        statepath["keys"]=["targetOutputSeq","thisstate","begin","end","localLen","dataNames","nextstate","localNlpOut","localNlpTrans","viterbiprob","sumprob"]
        statepath["key2Idx"] = dict( (name, idx) for (idx, name) in enumerate(statepath["keys"]) )
        statepath["values"] = []
        
        while thisstate!="LAST":
            (viterbiprob, localLen, viterbichoice, localNlpOut, localNlpTrans, sumprob) = self.memo[self.key(thisstate,begin,end)]
            statepath["values"].append( [self.targetOutputSeq, thisstate,begin,end, localLen, self.targetOutputNames[begin], viterbichoice, localNlpOut, localNlpTrans, viterbiprob, sumprob] )

            #### conusume localLen and move to next best state
            begin = begin+localLen
            thisstate = viterbichoice

        return(statepath)

    ################################
    def clearCounts( self ):
        # cycle through all hmm states and clear counts

        for ii in self.typeToIndex["HMMState"]:
            self.model[ii].counts = []

        for ii in self.typeToIndex["HMMStateOutputMVN"]:
            self.model[ii].countsData = []
            self.model[ii].countsPosterior = []

    ################################
    def addCountsFromViterbi( self, viterbiPath ):
        # Given viterbiPath fill out counts for transistions and emissions
        
        for vv in viterbiPath["values"]:
            #### have viterbi path transistioning from thisstate to nextstate with emission
            thisstate=vv[ viterbiPath["key2Idx"]["thisstate"] ]
            nextstate=vv[ viterbiPath["key2Idx"]["nextstate"] ]
            emission=vv[ viterbiPath["key2Idx"]["dataNames"] ]
            print("viterbi transistion emission", thisstate,nextstate, emission)

            hmmthisstate = self.name2state(thisstate)
            nextIdx = hmmthisstate.nextName2Idx[nextstate]
            counts = [0.0]*len(hmmthisstate.nexts)
            counts[nextIdx] = 1.0
            hmmthisstate.counts.append(counts)

            if hmmthisstate.probOut != "SILENT":
                outputthisstate = self.name2state( hmmthisstate.probOut )
                outputthisstate.countsData.append( emission )
                outputthisstate.countsPosterior.append( 1.0  )
                
    ################################
    def estimateModelFromCounts( self ):
        # estimate model from counts

        #### HMMState
        for ii in self.typeToIndex["HMMState"]:
            state = self.model[ii]

            ## add prior
            state.counts.append([1.0]*len(state.nexts))

            print("estimateModelFromCounts_counts", state.name, state.counts)
            ## sum counts
            npc = np.array(state.counts)
            sumCounts = np.sum(npc,axis=0)
            print("estimateModelFromCounts_sumCounts", sumCounts)
            probs = sumCounts/np.sum(sumCounts)
            print("estimateModelFromCounts_probs", probs)
            state.nextp = [ nl( xx ) for xx in probs ]

        for ii in self.typeToIndex["HMMStateOutputMVN"]:
            state = self.model[ii]
            print("estimateModelFromCounts_countsData", state.name, state.countsData)

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
    def forward( self, thisstate, begin=None, end=None):

        """returns the forward (Viterbi probability, best previous choice,
           summed all paths prob) of starting from START and deriving 
           [begin:end) while ending at thisstate. P(START->[begin,end) AND end at thisstate).
           Begin parameter is usually fixed at 0 for regular grammars.

        """

        #### fill in boundaries on first call
        if begin is None: begin=0
        if end is None: end=len(self.targetOutputNames)

        if self.key(thisstate,begin,end) in self.memoForward:
            return self.memoForward[self.key(thisstate,begin,end)]
        
        # at START
        if thisstate=="START":
            if (end-begin)==0:
                # START can derive empty string with prob=1.0
                result = (nl(1.0),0,"None",0,0,nl(1.0))
            else:
                # START cannot derive non-empty string directly, so prob=0.0
                result = (nl(0.0),0,"None",0,0,nl(0.0))
            self.memoForward[self.key(thisstate,begin,end)]=result
            return(result)
        
        # at LAST - it doesn't emit anything, but we need to compute how to reach it
        if thisstate=="LAST":
            # LAST doesn't emit anything - it just terminates
            localNlp = nl(1.0) # no emission cost
            localLen = 0       # doesn't consume anything
            
            # start with probability 0
            overallresult = (nl(0.0),0,"None",0,0,nl(0.0))
            Zsum = nl(0.0)
            
            # find all states that can transition to LAST
            prevStates = []
            prevProbs = []
            for i, state in enumerate(self.model):
                if hasattr(state, 'nexts'):  # this is an HMMState
                    for j, nextname in enumerate(state.nexts):
                        if nextname == "LAST":
                            prevStates.append(state.name)
                            prevProbs.append(state.nextp[j])
            
            # print(f"DEBUG: Forward LAST[{begin}:{end}], prevStates={prevStates}")
            
            # cycle through previous states that can reach LAST
            for ch in range(len(prevStates)):
                transNlp = prevProbs[ch]
                prevState = prevStates[ch]

                # recursive call: what's the forward probability of reaching the previous state
                # while deriving [begin, end) (same range since LAST doesn't consume)
                lhs = self.forward( prevState, begin, end)

                ViterbiNlp = lhs[0] + transNlp + localNlp
                Zsum = nladd(Zsum, lhs[-1] + transNlp + localNlp)

                # print(f"DEBUG:   LAST prev={prevState}, lhs={lhs[0]:.3f}, trans={transNlp:.3f}, local={localNlp:.3f}, total={ViterbiNlp:.3f}")

                if ViterbiNlp<overallresult[0]:
                    overallresult = (ViterbiNlp, localLen, prevState, localNlp, transNlp, nl(0.0))
            
            # put Zsum back in
            overallresult = (overallresult[0],overallresult[1],overallresult[2],overallresult[3],overallresult[4],Zsum)
            self.memoForward[self.key(thisstate,begin,end)]=overallresult
            return(overallresult)

        if (end-begin)<0:
            # impossible derivation
            result = (nl(0.0),0,"None",0,0,nl(0.0))
            self.memoForward[self.key(thisstate,begin,end)]=result
            return(result)

        this = self.name2state(thisstate)

        # print("forward exploring", this.name, begin, end)

        #### TODO: implement banding which requires constraints on states
            
        # start with probability 0
        overallresult = (nl(0.0),0,"None",0,0,nl(0.0))
        Zsum = nl(0.0)

        # determine what this state outputs locally
        if this.probOut != "SILENT":
            if (end-begin)<=0:
                # impossible derivation for NOT SILENT
                result = (nl(0.0),0,"None",0,0,nl(0.0))
                self.memoForward[self.key(thisstate,begin,end)]=result
                return(result)
            localNlp = self.name2state(this.probOut).nlp( self.targetOutput[end-1] )  # emit at end position
            localLen = 1
        else:
            localNlp = nl(1.0) # silent output nothing with prob 1.0
            localLen = 0

        # find all states that can transition to thisstate
        prevStates = []
        prevProbs = []
        
        # Special handling for LAST - it can be reached from any state that transitions to LAST
        if thisstate == "LAST":
            for i, state in enumerate(self.model):
                if hasattr(state, 'nexts'):  # this is an HMMState
                    for j, nextname in enumerate(state.nexts):
                        if nextname == "LAST":
                            prevStates.append(state.name)
                            prevProbs.append(state.nextp[j])
        else:
            # For regular states, find predecessors
            for i, state in enumerate(self.model):
                if hasattr(state, 'nexts'):  # this is an HMMState
                    for j, nextname in enumerate(state.nexts):
                        if nextname == thisstate:
                            prevStates.append(state.name)
                            prevProbs.append(state.nextp[j])

        # print(f"DEBUG: Forward {thisstate}[{begin}:{end}], localLen={localLen}, prevStates={prevStates}")

        # cycle through previous states that can reach thisstate
        for ch in range(len(prevStates)):

            # transition probability from previous state to thisstate
            transNlp = prevProbs[ch]
            prevState = prevStates[ch]

            # recursive call: what's the forward probability of reaching the previous state
            # while deriving [begin, end-localLen)
            lhs = self.forward( prevState, begin, end-localLen)

            ViterbiNlp = lhs[0] + transNlp + localNlp
            Zsum = nladd(Zsum, lhs[-1] + transNlp + localNlp)

            # print(f"DEBUG:   prev={prevState}, lhs={lhs[0]:.3f}, trans={transNlp:.3f}, local={localNlp:.3f}, total={ViterbiNlp:.3f}")

            if ViterbiNlp<overallresult[0]:
                # update viterbi, if nl is smaller, probability is larger
                overallresult = (ViterbiNlp, localLen, prevState, localNlp, transNlp, nl(0.0))
            #print("***",this.name,begin,end,"ch",ch,"ViterbiNlp",ViterbiNlp, "localNlp",localNlp,"transNlp",transNlp,"Zsum",Zsum,"overallresult",overallresult)
            
        # put Zsum back in after summing over all
        overallresult = (overallresult[0],overallresult[1],overallresult[2],overallresult[3],overallresult[4],Zsum)
        self.memoForward[self.key(thisstate,begin,end)]=overallresult
        return(overallresult)

    ################################
    def forwardAlign( self, thisstate, begin=None, end=None):
        """Output the Viterbi path once forward has been computed"""
        #### fill in boundaries on first call
        if begin is None: begin=0
        if end is None: end=len(self.targetOutputNames)

        statepath = {}
        statepath["keys"]=["targetOutputSeq","thisstate","begin","end","localLen","dataNames","prevstate","localNlpOut","localNlpTrans","viterbiprob","sumprob"]
        statepath["values"] = []
        
        # Build path by tracing backward from LAST to START
        path_states = []
        path_info = []
        current_state = thisstate
        current_begin = begin
        current_end = end
        
        # Trace backward from the final state to START
        while current_state != "START":
            (viterbiprob, localLen, viterbichoice, localNlpOut, localNlpTrans, sumprob) = self.memoForward[self.key(current_state, current_begin, current_end)]
            
            # For forward alignment, we need to determine the data position
            # The current state consumes localLen at the end of its range
            if current_state != "LAST":
                data_end_pos = current_end
                data_start_pos = current_end - localLen
                if localLen > 0:
                    data_names = self.targetOutputNames[data_start_pos:data_end_pos]
                else:
                    data_names = []
            else:
                data_names = []
                data_start_pos = current_end
            
            path_info.append([
                self.targetOutputSeq, 
                current_state, 
                current_begin, 
                current_end, 
                localLen, 
                data_names, 
                viterbichoice, 
                localNlpOut, 
                localNlpTrans, 
                viterbiprob, 
                sumprob
            ])
            
            # Move to the previous state
            current_state = viterbichoice
            current_end = current_end - localLen

        # Reverse the path to get it in forward order (START to LAST)
        path_info.reverse()
        statepath["values"] = path_info
        
        return(statepath)

    ################################
    def getTotalLikelihood(self):
        """
        Compute the total likelihood P(observation sequence) using forward algorithm.
        This is needed for normalizing posterior probabilities.
        """
        if not hasattr(self, 'targetOutput') or self.targetOutput is None:
            raise ValueError("No target output set. Call this after setting targetOutput.")
        
        # The total likelihood is the forward probability of reaching LAST 
        # while consuming the entire sequence [0, len(targetOutput))
        result = self.forward("LAST", 0, len(self.targetOutputNames))
        return result[-1]  # return the summed probability (Zsum)

    ################################
    def posteriorTransitionProb(self, from_state, to_state, begin=None, end=None):
        """
        Compute posterior expected probability of transitioning from state i to state j
        for all time steps in the sequence.
        
        Returns a list of probabilities, one for each valid transition time step.
        
        gamma_ij(t) = P(q_t = i, q_{t+1} = j | O, lambda)
                    = alpha_i(t) * a_ij * b_j(O_{t+1}) * beta_j(t+1) / P(O|lambda)
        
        where:
        - alpha_i(t) is forward probability of being in state i at time t
        - a_ij is transition probability from state i to j  
        - b_j(O_{t+1}) is emission probability of state j for observation at t+1
        - beta_j(t+1) is backward probability from state j at time t+1
        """
        
        if begin is None: begin = 0
        if end is None: end = len(self.targetOutputNames)
        
        # Get total likelihood for normalization
        total_likelihood = self.getTotalLikelihood()
        
        # Find transition probability from from_state to to_state
        from_state_obj = self.name2state(from_state)
        trans_prob = None
        for i, next_state in enumerate(from_state_obj.nexts):
            if next_state == to_state:
                trans_prob = from_state_obj.nextp[i]
                break
        
        if trans_prob is None:
            # No direct transition exists
            return []
        
        posterior_probs = []
        
        # For each time step where transition can occur  
        for t in range(begin, end-1):  # end-1 because we need t+1 to exist for transition
            # alpha_i(t): forward prob of reaching from_state having consumed [0,t+1)
            # This represents being in from_state after consuming symbol at position t
            alpha_i_t = self.forward(from_state, 0, t+1)[-1]  # use summed probability
            
            # beta_j(t+1): backward prob from to_state deriving [t+1,end)
            # This represents the probability from to_state for the remaining sequence
            beta_j_t_plus_1 = self.backward(to_state, t+1, end)[-1]  # use summed probability
            
            # For transitions, we don't include emission probability here because
            # the alpha and beta already account for emissions at their respective positions
            
            # Compute posterior: alpha_i(t) * a_ij * beta_j(t+1) / P(O|lambda)
            numerator = alpha_i_t + trans_prob + beta_j_t_plus_1
            posterior_nlp = numerator - total_likelihood  # subtract log of normalizing constant
            
            posterior_prob = probFromNlp(posterior_nlp)
            posterior_probs.append(posterior_prob)
        
        return posterior_probs

    ################################
    def posteriorEmissionProb(self, state, symbol_index, begin=None, end=None):
        """
        Compute posterior expected probability of state i emitting symbol t
        for all time steps in the sequence.
        
        Returns a list of probabilities, one for each time step.
        
        gamma_i(t) = P(q_t = i | O, lambda) 
                   = alpha_i(t) * beta_i(t) / P(O|lambda)
        
        For emission of specific symbol:
        Expected count of state i emitting symbol t = sum over all t where O_t = symbol of gamma_i(t)
        """
        
        if begin is None: begin = 0
        if end is None: end = len(self.targetOutputNames)
        
        # Get total likelihood for normalization
        total_likelihood = self.getTotalLikelihood()
        
        posterior_probs = []
        
        # For each time step where the state can emit
        for t in range(begin, end):
            # For emission probabilities, we need the probability of being in this state
            # at time t, which combines forward and backward probabilities
            
            # Skip special states that don't emit
            if state in ["START", "LAST"]:
                posterior_probs.append(0.0)
                continue
            
            state_obj = self.name2state(state)
            if state_obj.probOut == "SILENT":
                posterior_probs.append(0.0)
                continue
            
            # alpha_i(t): forward prob of reaching state having consumed [0,t]
            # For emission at position t, we want the forward prob ending at this state at time t
            if t == 0:
                alpha_i_t = self.forward(state, 0, t+1)[-1]  # consumed up to position t
            else:
                alpha_i_t = self.forward(state, 0, t+1)[-1]  # consumed up to position t
            
            # beta_i(t): backward prob from state deriving [t+1,end)  
            beta_i_t = self.backward(state, t+1, end)[-1]
            
            # Compute posterior: alpha_i(t) * beta_i(t) / P(O|lambda) 
            # The emission probability is already accounted for in alpha and beta
            numerator = alpha_i_t + beta_i_t
            posterior_nlp = numerator - total_likelihood
            
            posterior_prob = probFromNlp(posterior_nlp)
            posterior_probs.append(posterior_prob)
        
        return posterior_probs

################################

def testPosterior( myhmm, viterbiPath ):
    print("\n" + "="*60)
    print("TESTING POSTERIOR PROBABILITIES")
    print("="*60)

    # Test 1: Total likelihood
    print("\n1. Total Likelihood:")
    total_like = myhmm.getTotalLikelihood()
    print(f"Total likelihood (NLP): {total_like}")
    print(f"Total likelihood (Prob): {probFromNlp(total_like)}")

    # Test 2: Posterior transition probabilities
    print("\n2. Posterior Transition Probabilities:")

    # Find available transitions from the Viterbi path
    transitions_to_test = []
    for i in range(len(viterbiPath["values"])):
        row = viterbiPath["values"][i]
        from_state = row[1]  # thisstate
        to_state = row[6]    # nextstate
        if to_state != "None":
            transitions_to_test.append((from_state, to_state))

    # Remove duplicates
    transitions_to_test = list(set(transitions_to_test))

    for from_state, to_state in transitions_to_test[:3]:  # Test first 3 transitions
        print(f"\nTransition: {from_state} -> {to_state}")
        trans_probs = myhmm.posteriorTransitionProb(from_state, to_state)
        if trans_probs:
            print(f"  Posterior probabilities by time step: {trans_probs[:5]}...")  # Show first 5
            print(f"  Sum of posterior probabilities: {sum(trans_probs)}")
        else:
            print(f"  No valid transition found")

    # Test 3: Posterior emission probabilities  
    print("\n3. Posterior Emission Probabilities:")

    # Test emission probabilities for states that emit
    states_to_test = []
    for i in range(len(viterbiPath["values"])):
        row = viterbiPath["values"][i]
        state = row[1]  # thisstate
        if state not in states_to_test and state != "START" and state != "LAST":
            states_to_test.append(state)

    for state in states_to_test[:2]:  # Test first 2 emitting states
        print(f"\nState: {state}")
        # Test for first symbol index (0)
        emission_probs = myhmm.posteriorEmissionProb(state, 0)
        if emission_probs:
            print(f"  Posterior emission probabilities (symbol 0): {emission_probs[:5]}...")  # Show first 5
            print(f"  Sum of posterior probabilities: {sum(emission_probs)}")
        else:
            print(f"  No emissions computed")

    print("\n" + "="*60)
    
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

        myhmm.targetOutputSeq = "RB_06" # "RB_video_0"
        (myhmm.targetOutputNames, myhmm.targetOutput) = data.dataSeqMatrixFromName(myhmm.targetOutputSeq )

        result = myhmm.backward("START")
        print("result",result)

        viterbiPath = myhmm.backwardAlign("START")
        print("viterbiPath")
        print("\t".join(viterbiPath["keys"]))
        for vv in viterbiPath["values"]:
            print("\t".join([str(xx) for xx in vv]))

        myhmm.clearCounts()
        myhmm.addCountsFromViterbi( viterbiPath )

        myhmm.estimateModelFromCounts()
        print("model estimated")
        print(myhmm)
        
        
    if False:
        myhmm = HMM()
        myhmm.readJSON("/media/datastore/storestudio/workspace2022/hidden-markov-model/hmm/hmm_twoState.json")

        # get some data and run forward
        data = HMM_DAT.Data( "/media/datastore/storestudio/workspace2022/hidden-markov-model/hmm/hmm_data.json" )
        print("data.listDataSeq()",data.listDataSeq())

        myhmm.targetOutputSeq = "RB_video_0"
        (myhmm.targetOutputNames, myhmm.targetOutput) = data.dataSeqMatrixFromName(myhmm.targetOutputSeq )
        result = myhmm.backward("START")
        print("result",result)

        viterbiPath = myhmm.backwardAlign("START")
        print("viterbiPath")
        print("\t".join(viterbiPath["keys"]))
        for vv in viterbiPath["values"]:
            print("\t".join([str(xx) for xx in vv]))
        
        testPosterior(myhmm, viterbiPath)
        
if __name__ == "__main__":
    main()
