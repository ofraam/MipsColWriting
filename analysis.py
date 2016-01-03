'''
Created on Jan 2, 2016

@author: oamir
'''
from DocMIP_withAlgs import *
from MIP_algs import DegreeOfInterestMIPs

if __name__ == '__main__':
    myMip = readMIPfromFile("Reagan", 595)
    liveObjs = myMip.getLiveObjects()
    g = myMip.mip
    results = []
    for (n,d) in g.nodes(True):   
        if d['type']=='user':
            ngbrs = nx.neighbors(g, n)
            userName = myMip.nodeIdsToUsers[n]
            if userName.startswith("anon")==False:
                rankedObjs = myMip.rankLiveObjectsForUser(userName,alpha=0.0,beta=1.0,similarity="edge")
                i = 1;
                for o in rankedObjs:
#                     print userName
#                     print o
                    res = []
                    res.append(userName.encode('utf-8'))
                    res.append(str(o[0]))
                    res.append(str(o[1]))
                    res.append(str(i))
                    res.append(d['numRevisions'])
                    print userName+","+str(o[0])+","+str(o[1])+","+str(i)
                    i=i+1
                    results.append(res)
                    
    writeResultsToFile(results, "liveObjRanksReaganEdge.csv")
#                 print 'ranking for of '+myMip.nodeIdsToUsers[n]
#                 for ngb in ngbrs:
#                     print str(ngb)+","+str(g[n][ngb]['weight'])