import numpy as np
import math
import argparse
import copy

# all probabilities are stored at -log(p)
# https://gasstationwithoutpumps.wordpress.com/2014/05/06/sum-of-probabilities-in-log-prob-space/
def nl( prob ):
    if prob==0.0:
        #zero probabilties are special
        return(1.0E+300)
    else:
        return(-math.log(prob))

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
    return(nl1 + nl(1.0+math.exp(-nl2+nl1)))

################################
class HMMState:
    def __init__(self, inname="null", inprob=None):
        self.name = inname
        self.nexts = [] # next state
        self.nextp = [] # prob of next state
        self.prevs = [] # previous state that arrive here
        self.prevp = [] # prob previous state that arrive here
        self.prevComputed=False
        self.prob = [nl(pp) for pp in inprob] # 25 probabilities over HMM.label_symbols of output and "transistion"

        self.terminalToIdx = {}
        self.idxToTerminal = {}
        cc=0
        for insert in ["A","C","G","T",""]:
            for match in ["A","C","G","T",""]:
                key="%s%s" % (match,insert)
                self.terminalToIdx[key] = cc
                self.idxToTerminal[cc] = key
                cc+=1
        
        #     # degeneneracy delete+insert == ""+base never occurrs in alignment and should have prob~=0 at positions 4,9,14,19
        self.terminalToIdx[""] = 24
        self.idxToTerminal[24] = ""
        self.terminalToIdx["B"] = 25
        self.idxToTerminal[25] = "B"

        #print("self.terminalToIdx",self.terminalToIdx)
        
    #### allowed local output lengths
    def localLengths(self):
        if self.name=="last":
            return([0])
        else:
            #return([0,1,2])
            maxterminals = 10
            return( list(range(maxterminals)) )

    #### probability of local output. 
    def localProb( self, output ):

        """Can subtracts out local null model exp(-L) but I don't as contstant is arbitrary
        To account for insertion greater than 1:
        
        For single insertion, take path of Pnoextra = 0.999. viz
        insertions of two or more only happen 1/1000 times.

        For two+ insertions follow (1-Pnoextra) and then geometric
        with pgeo=0.9 mean=1.1
        """
        
        Pnoextra = nl(0.99)
        Pextra = nl(1-0.99)

        if len(output)<2:
            return(self.prob[self.terminalToIdx[output]] )
        elif len(output)==2:
            return(self.prob[self.terminalToIdx[output]] + Pnoextra)
        else:
            # first two are accordning to self.prob
            # additional is geometric p, q*p, q^2*p, ... mean= 1/p
            # can estimate by number of times missed bases in feature generation and number of missed (two stage)

            first = self.prob[self.terminalToIdx[output[0:2]]]
            pgeo=0.9 # mean=1.1
            geolen = len(output)-2
            # geo = (1-p)^(geolen-1) *p
            # log(geo) = (geolen-1)*log(1-p) + log(p)
            geo = -( (geolen-1)*math.log(1.0-pgeo) + math.log(pgeo) )

            return( Pextra + first + geo)
            
    def __str__(self):
        tmp=[]
        tmp.append("################")
        tmp.append("%s" % (self.name))
        tmp.append("nexts: %s" % ",".join([str(xx) for xx in self.nexts]))
        tmp.append("nextp: %s" % ",".join([str(xx) for xx in self.nextp]))
        tmp.append("prevs: %s" % ",".join([str(xx) for xx in self.prevs]))
        tmp.append("prevp: %s" % ",".join([str(xx) for xx in self.prevp]))
        tmp.append("prevComputed: %s" % str(self.prevComputed))
        if self.name!="last":
            Zsum = nl(0.0)
            for ii in range(len(self.prob)):
                if len(self.idxToTerminal[ii])==1 and ii<20: continue # degeneracy of delete+insert. multiple ways to derive singleton
                tmp.append("%s\t%g\t%s\t%d\tprob" % (self.idxToTerminal[ii], self.prob[ii], self.name, ii))
                Zsum = nladd(Zsum, self.prob[ii])
            tmp.append("Zsum: %f" % Zsum)
        return("\n".join(tmp))

################################        
class HMM:
    def __init__( self, probsLby25, hmmname="null" ):
        self.hmmname = hmmname
        self.state = []
        self.nameToStateIndex = {}
        self.memo = {}
        self.memoForward = {}
        self.targetOutput = "null"
        self.band = 256 # abs(thisstate-begin) cannot be larger than self.band
        
        # construct HMM based on probabilities
        for ii in range(probsLby25.shape[0]):
            inname = "state_%d_%s" % (ii,hmmname.decode())
            self.state.append( HMMState( inname, inprob=probsLby25[ii]) )
            # right now there is only one path
            self.state[ii].nexts.append((ii+1))
            self.state[ii].nextp.append(nl(1.0))
        self.state.append( HMMState(inname="last",inprob=[nl(1.0)]) )
        
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
                result = (nl(0.0),"",nl(0.0))
            else:
                result = (nl(1.0),"",nl(1.0))
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        if (end-begin)<0:
            #print "impossible derivation"
            result = (nl(0.0),"",nl(0.0))
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)

        if abs(thisstate-begin)>self.band:
            # limit large state deletions (thisstate>begin) or insertions (thisstate<begin)
            # assumes roughly linear relationship
            #print "impossible derivation"
            result = (nl(0.0),"",nl(0.0))
            self.memo[self.key(thisstate,begin,end)]=result
            return(result)
            
        # cycle through choices
        overallresult = (nl(0.0),"",nl(0.0))
        Zsum = nl(0.0)
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
            
            thisprob = localprob + transprob + rhs[0] # log p1*p2 = log p1 + log p2
            Zsum = nladd(Zsum, localprob+transprob+rhs[2]) 
            if thisprob<overallresult[0]:
                # if nl is smaller, probability is larger
                overallresult = (thisprob, thisoutput, nl(0.0))

        # put sum back in
        storeresult = (overallresult[0],overallresult[1],Zsum)
        # if self.state[thisstate+1].name=="last":
        #     print("last store",storeresult)
        self.memo[self.key(thisstate,begin,end)]=storeresult
        return(storeresult)

    ################################
    def outputToCigarOp(self, myoutput):
        if len(myoutput)==0: return("D")
        if len(myoutput)==1: return("M")
        if len(myoutput)>1:
            return("M"+"I"*(len(myoutput)-1) )
        
    ################################
    def backwardAlign( self, thisstate, begin, end):
        # print("backwardAlign",thisstate,begin,end)
        # name = self.state[thisstate].name
        # print("name",name)
        # if name=="last": return()
        # result = self.memo[self.key(thisstate,begin,end)]
        # print("result",result)
        # myprob = self.state[thisstate].localProb(result[1])
        # print("myprob",myprob)
        # self.backwardAlign( thisstate+1, begin+len(result[1]), end)

        # NOTE: the cigar string is with respect to the original
        # read. If a raw base has 99.999% delete probability and it is
        # used in the viterbi path is still counted as a Delete.

        statepath = {}
        statepath["keys"]=["thisstate","begin","end","name","outputprobPerEvent","outputprob","myoutput","oplist","viterbiprob","sumprob"]
        statepath["values"] = []
        
        prevOP = "*"
        prevOPcount = -1
        cigar = []
        name = self.state[thisstate].name
        while name!="last":
            (viterbiprob, myoutput, sumprob) = self.memo[self.key(thisstate,begin,end)]
            outputprob = self.state[thisstate].localProb(myoutput)
            oplist = self.outputToCigarOp(myoutput)
            numEvents = max(1,len(myoutput))
            outputprobPerEvent = outputprob/numEvents
            statepath["values"].append( [thisstate,begin,end,name, outputprobPerEvent, outputprob, myoutput, oplist, viterbiprob, sumprob] )

            for thisOP in oplist:
                if thisOP==prevOP:
                    prevOPcount = prevOPcount + 1
                else:
                    if prevOP!="*":
                        cigar.append("%d%s" % (prevOPcount,prevOP))
                    prevOP = thisOP
                    prevOPcount = 1
                
            thisstate = thisstate+1
            begin = begin+len(myoutput)
            name = self.state[thisstate].name

        # last cigar
        cigar.append("%d%s" % (prevOPcount,prevOP))
        statepath["cigar"] = cigar

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
    def productModelsTwo( self, modelB):
        """Update this to be the product this*modelB"""
        for ii in range(len(self.state)):
            if self.state[ii].name=="last": continue # no computation needed for dummy last state
            #### product is sum of costs
            for jj in range(len(self.state[0].prob)):
                costA = self.state[ii].prob[jj]
                costB = modelB.state[ii].prob[jj]
                print("productModelsTwo",ii,jj,costA,costB, costA+costB)
                self.state[ii].prob[jj] = costA + costB
            
    def productModels( listofmodels):
        """Take a product of HMM states"""
        #### for now assume all states are in correspondance and have the same length
        #### ambiguity will involve computing product distributions and simulanteous paths
        
        #### the resulting model starts with the first
        result = copy.deepcopy(listofmodels[0])
        for mm in range(1,len(listofmodels)):
            print("productModels", mm)
            result.productModelsTwo( listofmodels[mm] )
            #### TODO: show you can normalize at each step
            
        #### now result is the product (or sumCosts) of all the input models, now normalize
        #### Z=sum(-cost_i). Divide by Z and take log: -log(p_i/Z) = -log(p_i) - log(1/Z) = cost+log(Z) = cost - -log(Z)
        for ii in range(len(result.state)):
            if result.state[ii].name=="last": continue # no computation needed for dummy last state
            Z = nl(0.0) # 0.0
            for jj in range(len(result.state[0].prob)):
                Z= nladd(Z,result.state[ii].prob[jj]) # math.exp(-result.state[ii].prob[jj]) 
            print("productModels state Z",ii,Z)
            for jj in range(len(result.state[0].prob)):
                result.state[ii].prob[jj] += -Z # +math.log(Z)
        return(result)
    
################################
