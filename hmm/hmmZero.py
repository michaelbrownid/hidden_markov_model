import HMM
import math
import sys
import numpy as np

################################
# set up hmm
myhmm = HMM.HMM()

myref = "".join(["ACGT"]*(int(sys.argv[1])/4))

for ii in range(len(myref)):
    myhmm.pacbioMBSD("ref%d%s"%(ii,myref[ii]), myref[ii], (ii+1)*5)

last = HMM.HMMState()
last.name="last"
last.silent=True
myhmm.state.append(last)

print "length sumProb ELL ELL2 VAR"
ll=199
dd = myhmm.derive(0,ll)
print ll, dd[0],dd[1],dd[2], dd[2]-dd[1]*dd[1]

# for ll in range(0,200):
#     dd = myhmm.derive(0,ll)
#     print ll, dd[0],dd[1],dd[2], dd[2]-dd[1]*dd[1]

#print "-------------------"
#for (k,v) in  myhmm.memo.items():
#    print "%s %s" % (k,v)

