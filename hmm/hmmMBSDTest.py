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

myhmm.state.append("last")

# print "--------------------------------"
# print myhmm
# print "--------------------------------"
# print myhmm.generate()
# print "--------------------------------"

################################
trials = []
for ii in range(int(sys.argv[2])):
    tr = myhmm.generate()
    trials.append( math.log(tr["prob"]) )

print "%f\t%f\t%f" % (np.mean(trials), np.var(trials), np.std(trials))

