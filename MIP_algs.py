'''
Created on Mar 31, 2015

@author: Ofra
'''

'''
Compute degree of interest between user node and object node based on the given MIP
alpha: weight for apriori importance component (i.e., how important is obj in general in the MIP)
beta: weight for distance bsaed component (i.e., how close obj is to the user neighborhood)
'''

import networkx as nx

def DegreeOfInterestMIPs(mip, user, obj, alpha=0.3, beta=0.7):
    #compute apriori importance of node obj (considers effective conductance)
    current_flow_betweeness = nx.current_flow_betweenness_centrality(mip, True, 'weight');
    api_obj = current_flow_betweeness[obj]  #node centrality
    print 'obj'
    print obj
    print 'api_obj'
    print api_obj
    #compute proximity between user node and object node using Cycle-Free-Edge-Conductance from Koren et al. 2007
    cfec_user_obj = CFEC(user,obj,mip)
    print 'cfec_user_obj'
    print cfec_user_obj
    return alpha*api_obj+beta*cfec_user_obj
        
'''
computes Cycle-Free-Edge-Conductance from Koren et al. 2007
for each simple path, we compute the path probability (based on weights) 
'''
def CFEC(s,t,mip):
    R = nx.all_simple_paths(mip, s, t, cutoff=8)
    proximity = 0.0
    for r in R:
        PathWeight = mip.degree(r[0])*(PathProb(r,mip))  #check whether the degree makes a difference, or is it the same for all paths??
        proximity = proximity + PathWeight
    return proximity
    
        
def PathProb(path, mip):
    prob = 1.0
    for i in range(len(path)-1):
        prob = prob*(float(mip[path[i]][path[i+1]]['weight'])/mip.degree(path[i]))
    return prob
    
        

if __name__ == '__main__':
#    g = nx.Graph();
#    g.add_node(1)
#    g.add_node(2)
#    g.add_node(3)
#    g.add_node(4)
    attr = {}
    attr['weight']=1
#    
#    g.add_edge(1, 2,attr)
#    g.add_edge(1, 3,attr)
#    g.add_edge(2, 4,attr)
#    g.add_edge(3, 4,attr)
#    
#    print CFEC(1,4,g)
#    print DegreeOfInterestMIPs(g,1,2)
#    print DegreeOfInterestMIPs(g,1,4)
#    
    print '-------------'
    g2 = nx.Graph();
    g2.add_node(1)
    g2.add_node(2)
    g2.add_node(3)
    g2.add_node(4)
    g2.add_node(5)
    g2.add_edge(1, 2,attr)
   # g2.add_edge(1, 3,attr)
    g2.add_edge(2, 3,attr)
    g2.add_edge(3, 4,attr)
    #g2.add_edge(3, 5,attr)
    g2.add_edge(4, 5,attr)
    print CFEC(1,5,g2)
    print CFEC(1,2,g2)
    print CFEC(1,3,g2)
    print DegreeOfInterestMIPs(g2,1,2)
    print DegreeOfInterestMIPs(g2,1,3)
    print DegreeOfInterestMIPs(g2,1,4)
    print DegreeOfInterestMIPs(g2,1,5)
    
    

    
    