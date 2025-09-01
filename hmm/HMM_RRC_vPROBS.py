import numpy as np
import math
import argparse

# -log(0) for zero probabilties
nl0 = 1.0E+308

################################
class HMMState:
    def __init__(self, inname="null", inprob=None):
        self.name = inname
        self.nexts = [] # next state
        self.nextp = [] # prob of next state
        self.prevs = [] # previous state that arrive here
        self.prevp = [] # prob previous state that arrive here
        self.prevComputed=False
        self.prob = inprob # 25 probabilities over HMM.label_symbols of output and "transistion"

        self.terminalToIdx = {}
        self.idxToTerminal = {}
        cc=0
        for insert in ["a","c","g","t",""]:
            for match in ["A","C","G","T",""]:
                if match=="" and insert!="":
                    # degeneneracy delete+insert never occurrs in alignment and should have prob~=0
                    cc+=1
                    continue
                key="%s%s" % (match,insert)
                self.terminalToIdx[key] = cc
                self.idxToTerminal[cc] = key
                cc+=1
        self.terminalToIdx["B"] = 25
        self.idxToTerminal[25] = "B"
            
    #### allowed local output lengths
    def localLengths(self):
        if self.name=="last":
            return([0])
        else:
            return([0,1,2])

    #### probability of local output
    def localProb( self, output ):
        return(self.prob[self.terminalToIdx[output]])
    
    def __str__(self):
        tmp=[]
        tmp.append("################")
        tmp.append("%s" % (self.name))
        tmp.append("nexts: %s" % ",".join([str(xx) for xx in self.nexts]))
        tmp.append("nextp: %s" % ",".join([str(xx) for xx in self.nextp]))
        tmp.append("prevs: %s" % ",".join([str(xx) for xx in self.prevs]))
        tmp.append("prevp: %s" % ",".join([str(xx) for xx in self.prevp]))
        tmp.append("prevComputed: %s" % str(self.prevComputed))
        tmp.append("prob: %s" % ",".join([str(xx) for xx in self.prob]))
        return("\n".join(tmp))

################################        
class HMM:
    def __init__( self, probsLby25 ):
        self.state = []
        self.nameToStateIndex = {}
        self.memo = {}
        self.memoForward = {}
        self.targetOutput = "null"

        # construct HMM based on probabilities
        for ii in range(probsLby25.shape[0]):
            self.state.append( HMMState( inname="state"+str(ii), inprob=probsLby25[ii]) )
            # right now there is only one path
            self.state[ii].nexts.append((ii+1))
            self.state[ii].nextp.append(1.0)
        self.state.append( HMMState(inname="last",inprob=[1.0]) )
        
    def key(self, state,length):
        return("%s-%s" % (str(state),str(length)))

    def key(self, state,begin,end):
        return("%s-%s-%s" % (str(state),str(begin),str(end)))

    ################################
    def __str__(self):
        tmp=[]
        for ii in range(len(self.state)):
            tmp.append("%d %s" % (ii, str(self.state[ii])))
        return("\n".join(tmp))

    ################################
    def generate( self ):
        """Generate random strings from the HMM"""
        res = {}
        res["prob"]= 0.0
        #res["seq"]=[]

        prob=1.0
        here = 0
        while here != (len(self.state)-1):
            this = self.state[here]
            if  not this.silent:
                #print "output"
                ch = np.random.choice(len(this.out), p=this.out)
                prob=prob*this.out[ch]
                #res["seq"].append( (self.alphabet[ch], this.name, prob) )
            else:
                #res["seq"].append( ( "-", this.name, prob) )
                pass

            #print "trans"
            ch = np.random.choice(len(this.nextp), p=this.nextp)
            prob=prob*this.nextp[ch]
            here = this.nexts[ch]
            #print "here", here

        res["prob"] = prob
        return(res)

    ################################
    def backward( self, thisstate, begin, end):

        """returns the backward Vierbi probability, best choice, summed all
           paths prob of this state deriving [begin:end) of
           self.targetOutput.  P(thisstate->[begin,end)), the most
           natural language interpretation (inside). End is usually
           fixed at the end of the string for regular grammars.

        """

        if self.key(thisstate,begin,end) in self.memo:
            return self.memo[self.key(thisstate,begin,end)]

        this = self.state[thisstate]

        #print "exploring", this.name, begin, end

        # at end
        if this.name=="last":
            if (end-begin)!=0:
                result = (0.0,"",0.0)
            else:
                result = (1.0,"",1.0)
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        if (end-begin)<0:
            #print "impossible derivation"
            result = (0.0,"",0.0)
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        # cycle through choices
        overallresult = (-1.0,"",-1)
        Zsum = 0.0
        for outputLen in this.localLengths():
            # the next state begin point
            nextbegin = begin+outputLen
            thisoutput = self.targetOutput[begin:nextbegin]
            # output a outputLen terminal symbols with given probability
            localprob = this.localProb(thisoutput)
            # transistion right now is always single 1.0
            transprob = this.nextp[0]
            # continue the derivation
            rhs = self.backward( this.nexts[0], nextbegin, end)
            # if self.state[thisstate+1].name=="last":
            #     print("***",Zsum,this.name, begin, thisoutput, localprob, transprob, rhs)
            
            thisprob = localprob*transprob*rhs[0]
            Zsum = Zsum + localprob*transprob*rhs[2]
            if thisprob>overallresult[0]:
                overallresult = (thisprob, thisoutput,-1)

        storeresult = (overallresult[0],overallresult[1],Zsum)
        # if self.state[thisstate+1].name=="last":
        #     print("last store",storeresult)
        self.memo[self.key(thisstate,begin,end)]=storeresult
        return(storeresult)

    ################################
    def backwardAlign( self, thisstate, begin, end):
        print("backwardAlign",thisstate,begin,end)
        name = self.state[thisstate].name
        print("name",name)
        if name=="last": return()
        result = self.memo[self.key(thisstate,begin,end)]
        print("result",result)
        myprob = self.state[thisstate].localProb(result[1])
        print("myprob",myprob)
        self.backwardAlign( thisstate+1, begin+len(result[1]), end)
        

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
