import numpy as np
import math
import argparse
import copy
import json
import sys


################################
class DataPoint:
    def __init__(self, json):
        self.name = None
        self.data = None
        self.readJSON( json )

    def __str__(self):
        tmp=[]
        tmp.append('"type": "DataPoint"')
        tmp.append('"name": "%s"' % self.name)
        tmp.append('"data":  %s' % json.dumps(self.mean))
        return('{\n%s\n}\n' % ",\n".join(tmp))

    def readJSON(self, json):
        if json["type"] != "DataPoint": print("ERROR", json, "is not DataPoint")
        self.name = json["name"]
        self.data = json["data"]

################################
class DataSeq:
    def __init__(self, json):
        self.name = None
        self.data = None
        self.readJSON( json )

    def __str__(self):
        tmp=[]
        tmp.append('"type": "DataSeq"')
        tmp.append('"name": "%s"' % self.name)
        tmp.append('"data":  %s' % json.dumps(self.data))
        return('{\n%s\n}\n' % ",\n".join(tmp))

    def readJSON(self, json):
        if json["type"] != "DataSeq": print("ERROR", json, "is not DataSeq")
        self.name = json["name"]
        self.data = json["data"]

################################        
class Data:
    def __init__( self, injasonfile ):
        self.jsonfile = injasonfile
        self.json = None
        self.nameToIndex = {}
        self.typeToIndex = {}

        self.readJSON( self.jsonfile)
        
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

    ################################
    def __str__(self):
        #### output string of list of json states based on the model states
        tmp=[]
        for ii in range( len(self.model) ):
            this = self.model[ii]
            tmp.append(str(this))
        return("[\n%s\n]\n" % ",\n".join(tmp))

    ################################
    def listDataSeq( self ):
        return( [ self.json[ ii ]["name"] for ii in self.typeToIndex["DataSeq"] ] )

    ################################
    def dataSeqMatrixFromName( self, myname ):
        # return data[Time][n-dim]
        dataNames = []
        data = []
        for datapointname in self.json[self.nameToIndex[myname]]["data"]:
            dataNames.append(datapointname)
            data.append( self.json[ self.nameToIndex[ datapointname ] ]["data"] )
        name2Idx = dict( (name,idx) for (idx,name) in enumerate(dataNames) )
        return((dataNames,data,name2Idx))

