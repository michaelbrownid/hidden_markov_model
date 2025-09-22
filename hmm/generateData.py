import sys

""" Generate data that looks like:
[
{
    "type": "DataPoint",
    "name": "RB_0_S_0",
    "data": [ -4, 3, -2 ]
}
,
{
    "type": "DataPoint",
    "name": "RB_0_S_1",
    "data": [ 3, -2, 4 ]
}
,
{
    "type": "DataSeq",
    "name": "RB_video_0",
    "data": [ "RB_0_S_0", "RB_0_S_1" ]
}
]
"""

def dataTSVToJSON( filenameTSV, prefix ):

    model = []
    names = []
    dat = open(filenameTSV).read().splitlines()
    
    for ii in range(len(dat)):

        tmp = []
        tmp.append('"type": "DataPoint"')
        name = "%s_S_%d" % (prefix,ii)
        names.append(name)
        tmp.append('"name": "%s"' % name)

        data = dat[ii].split("\t")
        tmp.append('"data": [ %s ]' % ", ".join(data))
        
        model.append('{\n%s\n}\n' % ",\n".join(tmp))
        
    #### Genereate DataDdeq
    tmp = []
    tmp.append('"type": "DataSeq"')
    tmp.append('"name": "%s"' % prefix)
    tmp.append('"data": [ "%s" ]' % '", "'.join(names))

    model.append('{\n%s\n}\n' % ",\n".join(tmp))

    return('[\n%s\n]\n' % ",\n".join(model))

if __name__ == "__main__":
    print(dataTSVToJSON(sys.argv[1], sys.argv[2]))
