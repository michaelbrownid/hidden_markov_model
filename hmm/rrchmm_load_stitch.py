"""load rrchmm, features from .h5 file and stitch to get resulting
linear model without windows and overlaps.

For example: /home/UNIXHOME/mbrown/mbrown/workspace2022Q1/rrchmm:
  -rw-rw-rw-  1 mbrown Domain Users  23349248 2022.03.05_09:48 RRCFeaturesZWM98.corrected.array.h5
  -rw-rw-rw-  1 mbrown Domain Users   9685072 2022.03.05_09:41 RRCFeaturesZWM98-ref-subread.align.allsub.h5

"""

import rawcorrgpu.stitcher
import h5py
import numpy as np

class loadStitch:

    def __init__(self):
        self.predictionsFile = None
        self.featuresFile = None
        self.mapping = {}
        self.label_symbols = {}
        cc=0
        for insert in ["a","c","g","t",""]:
            for match in ["A","C","G","T",""]:
                #label_symbols.append("%s%s" % (match,insert))
                self.label_symbols[cc] = "%s%s" % (match,insert)
                cc+=1
        self.label_symbols[25]="" # burst
        
    def computeStitch( self, predictionsFile, featuresFile ):

        self.predictionsFile = predictionsFile
        self.featuresFile = featuresFile

        self.predictions = h5py.File(predictionsFile,"r")
        self.features = h5py.File(featuresFile,"r")

        read_ids = self.features["read_ids"]
        all_pos = self.features["positions"]

        self.read_intervals = rawcorrgpu.stitcher.computeReadIntervals( read_ids )
        for begin, end in self.read_intervals:
            read_id = read_ids[begin] # identical across range
            stitched = rawcorrgpu.stitcher.stitch( all_pos[begin:end])
            for index in range(len(stitched)):
                (rangebegin, rangeend) = stitched[index]
                basebegin = all_pos[begin+index,rangebegin]
                baseend = all_pos[begin+index,rangeend-1]+1 # take included base and add one to get open-right-end

                # this read_id has list of windows. Window (begin+index) contributes window positions (rangebegin:rangeend) correspondning to read bases (basebegin:baseend)
                if read_id not in self.mapping:
                    # print("# computeStitch adding", read_id)
                    self.mapping[read_id] = []
                mydat = (begin+index, rangebegin,rangeend, basebegin[0], baseend[0])
                #print("mydat",read_id,mydat)
                self.mapping[read_id].append ( mydat )

    def getReadPredictions( self, read_id ):
        mypreds = []
        for dd in self.mapping[ read_id ]:
            mywindow = dd[0]
            mybegin = dd[1]
            myend = dd[2]
            mybasebegin=dd[3]
            mybaseend=dd[4]
            #print("getReadPredictions", read_id, dd)
            mypreds.append( self.predictions["all_preds"][mywindow,mybegin:myend].squeeze()) # 25 probabilities
        return(np.concatenate(mypreds)) # TODO: correct?
    
    def getReadBaseFeatures( self, read_id ):
        mypreds = []
        for dd in self.mapping[ read_id ]:
            mywindow = dd[0]
            mybegin = dd[1]
            myend = dd[2]
            mybasebegin=dd[3]
            mybaseend=dd[4]
            #print("getReadBaseFeatures", read_id, dd)
            mypreds.append( self.features["features"][mywindow,mybegin:myend,0:4].squeeze()) # 4 one-hot encoding ACGT
        return(np.concatenate(mypreds)) # TODO: correct?



