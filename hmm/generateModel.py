import numpy as np

""" Generate a model that looks like:
    [
    {
	"type": "HMMStateOutputMVN",
	"name": "RBS_0_out",
	"mean": [ -4, 3, -2 ],
	"sd": [10,10,10 ]
    },
    {
	"type": "HMMStateOutputMVN",
	"name": "RBS_1_out",
	"mean": [ 3, -2, 4 ],
	"sd": [ 8,8,8 ]
    },
    {
	"type": "HMMState",
	"name": "RBS_0",
	"next": [ ["RBS_1",0.9], ["RBS_0",0.05 ], ["LAST",0.05 ]],
	"probOut": "RBS_0_out"
    },
    {
	"type": "HMMState",
	"name": "RBS_1",
	"next": [ ["LAST", 0.7], ["RBS_1",0.15], ["RBS_0",0.15 ] ],
	"probOut": "RBS_1_out"
    },
    {
	"type": "HMMState",
	"name": "START",
	"next": [ ["RBS_0",0.8], ["RBS_1",0.2] ],
	"probOut": "SILENT"
    }
]
"""

def almostUniform( num ):
    alpha = np.ones(num) * 10000.0  # higher = closer to uniform, lower = spikier
    p = np.random.dirichlet(alpha)
    #print(p)
    #print("Sum:", p.sum())
    return(p)

def genNext( nn ):
    # [ ["LAST", 0.7], ["RBS_1",0.15], ["RBS_0",0.15 ] ]
    probs = almostUniform(nn+1) # plus LAST
    tmp = []
    for ii in range(nn):
        name = "RBS_%d" % ii
        tmp.append('["%s",%f]' % (name,probs[ii]))
    name = "LAST"
    tmp.append('["%s",%f]' % (name,probs[-1]))
    return('[ %s ]' % ", ".join(tmp))
        
def main():

    num = 6
    model = []
    
    #### Generate output = mean SD
    dat = open("kmeans.dat").read().splitlines()

    ii = 0
    while ii<len(dat):
        tmp = []
        (idx,type,data) = dat[ii].split(",",2)
        tmp.append('"type": "HMMStateOutputMVN"')
        
        name = "RBS_%d_out" % int(ii/2)
        tmp.append('"name": "%s"' % name)

        #### replace kmeans with normal random
        #samplenorm = np.random.normal(loc=0,scale=1.0, size=1024)
        #data = ",".join([str(xx) for xx in samplenorm])

        tmp.append('"mean": [ %s ]' % data)

        # Now do the sd
        
        ii +=1
        (idx,type,data) = dat[ii].split(",",2)

        ### increase SD
        # sdAdd = 2.0
        # datav = data.split(",")
        # datavp = [float(xx)+sdAdd for xx in datav]
        # data = ",".join([str(xx) for xx in datavp])
        
        #### replace kmeans sd with constant sd
        # samplenorm = [2.0]*1024
        # data = ",".join([str(xx) for xx in samplenorm])
        
        tmp.append('"sd": [ %s ]' % data)

        model.append('{\n%s\n}\n' % ",\n".join(tmp))

        ii +=1

    #### Genereate state transistions
    for nn in range(num):
        tmp = []
        tmp.append('"type": "HMMState"')
        tmp.append('"name": "RBS_%d"' % nn)
        tmp.append('"probOut": "RBS_%d_out"' % nn)
        tmp.append('"next": %s' % genNext( num ))
        model.append('{\n%s\n}\n' % ",\n".join(tmp))

    tmp = []
    tmp.append('"type": "HMMState"')
    tmp.append('"name": "START"')
    tmp.append('"probOut": "SILENT"')
    tmp.append('"next": %s' % genNext( num ))
    model.append('{\n%s\n}\n' % ",\n".join(tmp))

    print('[\n%s\n]\n' % ",\n".join(model))

main()    
