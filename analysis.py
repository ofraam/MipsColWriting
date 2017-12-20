'''
Created on Jan 2, 2016

@author: oamir
'''
from DocMIP_withAlgs import *
from MIP_algs import DegreeOfInterestMIPs
import csv

def mergeResults(folderName="results"):
    mergedResults=[]
    for a in os.walk(folderName):
        for csvFile in a[2]:
            shortFile = csvFile[:-4]
            fullPath = os.path.join(os.getcwd(), folderName, csvFile)
            with open(fullPath, 'rb') as csvfile:
                
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    if ((row[9]=='3_precision')| (row[9]=='5_precision')):
                        row.append(shortFile)
                        mergedResults.append(row)
    writeResultsToFile(mergedResults,"mergedResults25.csv")
                    
if __name__ == '__main__':
#     mergeResults()
    myMip = readMIPfromFile("Film_noir", 464)
    liveObjs = myMip.getLiveObjects()
    g = myMip.mip
    results = []
    for (n,d) in g.nodes(True):   
        if d['type']=='user':
            ngbrs = nx.neighbors(g, n)
            userName = myMip.nodeIdsToUsers[n]
            if userName.startswith("anon")==False:
                rankedObjs = myMip.rankLiveObjectsForUser(userName,alpha=0.0,beta=1.0,similarity="adamic")
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
                     
    writeResultsToFile(results, "liveObjRanksFilm_noirAdamic.csv")
#                 print 'ranking for of '+myMip.nodeIdsToUsers[n]
#                 for ngb in ngbrs:
#                     print str(ngb)+","+str(g[n][ngb]['weight'])
  
  
                  