from . import HMM_RRC
import argparse
import numpy as np

################################

def main(args):

    print("================================")
    if True:
        # read in probs and construct HMM
        probs = np.loadtxt(args.nptxt) 
        myhmm = HMM_RRC.HMM( probs )
        print("myhmm",myhmm)

        """
     **                       *                  *                                  *
    T....G....A....G....A....T....G....G....G....T....T....C....T....T....G....C....A....C....T....G.... Ref reference_+	A	3	C	
    T....G....A....G....A....T....g....G....G....G....T....C....T....T....G....C....C....C....T....G.... +	 m64011_181218_235052/98/8679_20388_+	
    T....G....A....G....A....T....-....G....G....T....T....C....T....T....G....C....C....C....T....G.... m64011_181218_235052/98/8681_20686_+   ORIGINAL
        """


        # truth
        myhmm = HMM_RRC.HMM( probs ) # init memo
        myhmm.targetOutput = "TGAGATGGGTTCTTGCACTG"
        print(">>>>> TRUTH",myhmm.targetOutput)
        print(myhmm.backward(0,0,len(myhmm.targetOutput)))
        myhmm.backwardAlign(0,0,len(myhmm.targetOutput))

        # RRC
        myhmm = HMM_RRC.HMM( probs ) # init memo
        myhmm.targetOutput = "TGAGATGGGGTCTTGCCCTG"
        print(">>>>> RRC",myhmm.targetOutput)
        print(myhmm.backward(0,0,len(myhmm.targetOutput)))
        myhmm.backwardAlign(0,0,len(myhmm.targetOutput))

        # original
        myhmm = HMM_RRC.HMM( probs ) # init memo
        myhmm.targetOutput = "TGAGATGGTTCTTGCCCTG"
        print(">>>>> ORGINAL",myhmm.targetOutput)
        print(myhmm.backward(0,0,len(myhmm.targetOutput)))
        myhmm.backwardAlign(0,0,len(myhmm.targetOutput))

    print("================================")
    if True:

        for seq in ["TGAGATGGGTTCTTGCACTG","TGAGATGGGGTCTTGCCCTG","TGAGATGGTTCTTGCCCTG"]:
            print("================",seq)
            myhmm = HMM_RRC.HMM( np.loadtxt("subregion.98.8679.preds.txt") )
            myhmm.targetOutput = seq
            print(">>>>> 8679",myhmm.targetOutput)
            res = myhmm.backward(0,0,len(myhmm.targetOutput))
            print(res)
            p1 = res[0]

            myhmm = HMM_RRC.HMM( np.loadtxt("subregion.98.129456.preds.txt") )
            myhmm.targetOutput = seq
            print(">>>>> 129456",myhmm.targetOutput)
            res = myhmm.backward(0,0,len(myhmm.targetOutput))
            print(res)
            p2 = res[0]

            myhmm = HMM_RRC.HMM( np.loadtxt("subregion.98.201503.preds.txt") )
            myhmm.targetOutput = seq
            print(">>>>> 201503",myhmm.targetOutput)
            res = myhmm.backward(0,0,len(myhmm.targetOutput))
            print(res)
            p3 = res[0]
            myhmm.backwardAlign(0,0,len(myhmm.targetOutput))
            print("===",seq,p1*p2*p3)


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nptxt")
    args=parser.parse_args()
    main(args)

