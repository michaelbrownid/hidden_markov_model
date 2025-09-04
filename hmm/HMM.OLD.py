import numpy as np
import math

################################
class HMMStateOutput:
    # HMM state output probabilities. return probability of state outputting given symbols

    #### define for DNA
    def __init__(self):
        self.parameters = [0.25, 0.25, 0.25, 0.25
        
    
################################
class HMMState:
    def __init__(self):
        self.name = "HMMState"
        self.silent = False
        self.out = []   # emisson prob
        self.nexts = [] # next state
        self.nextp = [] # prob of next state
        self.targetOutput = None
        self.prevs = [] # previous state that arrive here
        self.prevp = [] # prob previous state that arrive here
        self.prevComputed=False

    def __str__(self):
        tmp=[]
        tmp.append("################")
        tmp.append("%s silent:%s" % (self.name, self.silent))
        tmp.append("out: %s" % ",".join([str(xx) for xx in self.out]))
        tmp.append("nexts: %s" % ",".join([str(xx) for xx in self.nexts]))
        tmp.append("nextp: %s" % ",".join([str(xx) for xx in self.nextp]))
        tmp.append("prevs: %s" % ",".join([str(xx) for xx in self.prevs]))
        tmp.append("prevp: %s" % ",".join([str(xx) for xx in self.prevp]))
        tmp.append("prevComputed: %s" % str(self.prevComputed))
        return("\n".join(tmp))

################################        
class HMM:
    def __init__( self ):
        self.state = []
        self.nameToStateIndex = {}
        self.alphaToInd = {"A":0, "C":1, "G":2, "T":3}
        self.memo = {}
        self.memoForward = {}

    def key(self, state,length):
        return("%s-%s" % (str(state),str(length)))

    def key(self, state,begin,end):
        return("%s-%s-%s" % (str(state),str(begin),str(end)))

    ################################
    def mylog(self, xx):
        if (xx<=0.0):
            return(99E+99)
        else:
            return(math.log(xx))

    def emLL(self, xx):
        tmp = 0.0
        for tt in xx:
            if tt>0.0: tmp=tmp+tt*math.log(tt)
        return(tmp)

    def emLL2(self, xx):
        tmp = 0.0
        for tt in xx:
            if tt>0.0: tmp=tmp+tt*math.log(tt)*math.log(tt)
        return(tmp)

    ################################
    def __str__(self):
        tmp=[]
        for ii in range(len(self.state)):
            tmp.append("%d %s" % (ii, str(self.state[ii])))
        return("\n".join(tmp))

    ################################
    def setSize( self, inSize ):
        self.state = [ HMMState() for ii in range(inSize)]

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
    def pacbioMBSD( self, prefix, toIndex, ctx, params ):
        """
Generate 5 state sub-hmm representing Match Branch Stick Delete for one base:
0=Begin
1=Match
2=Branch
3=Stick
4=Delete
        """

        (pm,pb,ps,pd) = params.ftrans(ctx)

        thisInd = len(self.state)

        this = HMMState()
        this.name = prefix+"_begin"
        this.silent = True
        this.nexts.append(thisInd+1)
        this.nextp.append(1.0*pm)
        this.nexts.append(thisInd+2)
        this.nextp.append(1.0*pb)
        this.nexts.append(thisInd+3)
        this.nextp.append(1.0*ps)
        this.nexts.append(thisInd+4)
        this.nextp.append(1.0*pd)
        self.state.append(this)

        this = HMMState()
        this.name = prefix+"_match"
        this.silent = False
        this.nexts.append(toIndex)
        this.nextp.append(1.0)
        this.out = params.foutm(ctx)
        self.state.append(this)

        this = HMMState()
        this.name = prefix+"_branch"
        this.silent = False
        this.nexts.append(thisInd+1)
        this.nextp.append(1.0*pm)
        this.nexts.append(thisInd+2)
        this.nextp.append(1.0*pb)
        this.nexts.append(thisInd+3)
        this.nextp.append(1.0*ps)
        this.nexts.append(thisInd+4)
        this.nextp.append(1.0*pd)
        this.out = params.foutb(ctx)
        self.state.append(this)

        this = HMMState()
        this.name = prefix+"_stick"
        this.silent = False
        this.nexts.append(thisInd+1)
        this.nextp.append(1.0*pm)
        this.nexts.append(thisInd+2)
        this.nextp.append(1.0*pb)
        this.nexts.append(thisInd+3)
        this.nextp.append(1.0*ps)
        this.nexts.append(thisInd+4)
        this.nextp.append(1.0*pd)
        this.out = params.fouts(ctx)
        self.state.append(this)

        this = HMMState()
        this.name = prefix+"_delete"
        this.silent = True
        this.nexts.append(toIndex)
        this.nextp.append(1.0)
        self.state.append(this)

################################
    def derive( self, thisstate, length):

        """returns the (totalProb, EXP(LL), and EXP[LL^2]) for thisstate deriving length string"""

        if self.key(thisstate,length) in self.memo:
            return self.memo[self.key(thisstate,length)]

        this = self.state[thisstate]

        # at end
        if this.name=="last":
            result = (1.0, 0.0, 0.0)
            return(result)

        if this.silent:
            newlength = length
            expEmLL  = 0.0
            expEmLL2 = 0.0
        else:
            newlength = length-1
            expEmLL = self.emLL(this.out)
            expEmLL2 = self.emLL2(this.out)

        # would require impossible derivation
        if newlength<0:
            result = (0.0, 99E+99, 99E+99)
            self.memo[self.key(thisstate,length)]=result
            return(result)

        # cycle through choices
        overallresult = [0.0, 0.0, 0.0]
        for ch in range(len(this.nexts)):

            transp = this.nextp[ch]
            rhs = self.derive( this.nexts[ch], newlength)

            A = self.mylog(transp) # log transition
            A2 = A*A               # log transistion squared
            B = expEmLL            # expected log emission likelihood
            B2= expEmLL2           # expected log squared emission likelihood
            C = rhs[1]             # EXP[LL] of next
            C2= rhs[2]             # EXP[LL^2] of next
            thisll =  transp*(A+B+C)
            thisll2 = transp*( A2 + B2 + C2 + 2*A*B + 2*A*C +2*B*C)

            thisprob = transp*rhs[0]

            if thisprob>0.0:
                overallresult[0]=overallresult[0]+thisprob
                overallresult[1]=overallresult[1]+thisll
                overallresult[2]=overallresult[2]+thisll2

        self.memo[self.key(thisstate,length)]=overallresult
        return(overallresult)

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
                result = (0.0,-1,0.0)
            else:
                result = (1.0,-1,1.0)
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        if (end-begin)<0:
            #print "impossible derivation"
            result = (0.0,-1,0.0)
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        if not this.silent:
            if (end-begin)<1:
                #print "impossible for non-silent"
                result = (0.0,-1,0.0)
                self.memo[self.key(thisstate,begin,end)]=result
                return(result)

            probEmission = this.out[self.alphaToInd[self.targetOutput[begin]]]
            nextbegin = begin+1
        else:
            probEmission = 1.0
            nextbegin= begin

        # cycle through choices
        overallresult = (-1.0,-1,-1)
        Zsum = 0.0
        for ch in range(len(this.nexts)):
            # this state make emssion, makes choice, and proceeds
            transp = this.nextp[ch]
            rhs = self.backward( this.nexts[ch], nextbegin, end)
            thisprob = probEmission*transp*rhs[0]
            Zsum = Zsum + probEmission*transp*rhs[2]
            if thisprob>overallresult[0]:
                overallresult = (thisprob, this.nexts[ch],-1)

        storeresult = (overallresult[0],overallresult[1],Zsum)
        self.memo[self.key(thisstate,begin,end)]=storeresult
        return(storeresult)

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
