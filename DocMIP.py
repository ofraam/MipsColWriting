'''
Created on Jan 25, 2015

@author: Ofra
'''

import version 
#from version import Version
#from version import Page
import networkx as nx
import sys 
import cPickle, copy, custom_texttiling as ct, difflib, gensim, jellyfish, matplotlib.pyplot as plt, matplotlib.cm as cm
import Levenshtein, nltk, nltk.data, numpy as np, os, re
from pylab import gca, Rectangle
from difflib import SequenceMatcher
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from matplotlib import rcParams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt





stop = stopwords.words('english')

class Session:
    def __init__(self, user, revision, time):
        self.actions = []
        self.user = user 
        self.time = time
        
class Action:
    def __init__(self, user, actType, desc, weightInc):
        self.user = user
        self.actType = actType
        self.desc = desc
        self.weightInc = weightInc
        
        

class Mip:
    def __init__(self, firstVersion):
        self.mip = nx.Graph()
        self.pars = []
        self.users = {}
        self.nodeIdsToUsers = {}
        self.latestVersion = 0
        self.lastID = 0
        self.decay = 0.01
        self.sigIncrement = 1
        self.minIncrement = 0.1
        self.currentVersion = firstVersion 
        

    def initializeMIP(self):
        self.pars.append({})
        userId = self.addUser(self.currentVersion.author)
#        print userId
        index = 0
        for par in self.currentVersion.paragraphs:
            parId = self.addPars(None, index)
#            print parId
            self.updateEdge(userId, parId,'u-p', self.sigIncrement)
            index=index+1
            
        for i in range(0,len(self.currentVersion.paragraphs)):
            for j in range(i+1,len(self.currentVersion.paragraphs)):
                if i!=j:
                    self.updateEdge(self.pars[self.latestVersion-1][i], self.pars[self.latestVersion][j],'p-p', self.sigIncrement)
                    
        
#        print self.mip.nodes(True)
        
        
                
    def updateMIP(self, newVersion):
        self.pars.append({})
        self.latestVersion=self.latestVersion+1
        #clear updated edges
        for edge in self.mip.edges_iter(data=True):
            edge[2]['updated']=0
            
        #get user
        userId = self.addUser(newVersion.author)
            
      
#        print len(newVersion.paragraphs)
        (new_old_mappings,old_new_mappings) = generate_mapping_for_revision(self.currentVersion,newVersion)
        mappings=new_old_mappings
#        print mappings
#        print self.currentVersion.paragraphs[5].text
#        print newVersion.paragraphs[4].text

        #check for significant changes, additions and deletions; add to MIP
        old_partext = [a.text.encode('utf-8') for a in self.currentVersion.paragraphs]
        new_partext = [a.text.encode('utf-8') for a in newVersion.paragraphs]
        sigChangePars=[]
        smallChangePars=[]
        addedPars=[]
        deletedPars=[] #note this is from *previous* revision

        for i in range(0,len(newVersion.paragraphs)):
#            print i
            if i in mappings: #node in MIP already exists, just update
#                print mappings[i]
                prevParIndex=mappings[i]
                self.addPars(prevParIndex, i) # adding to MIP
                #check for sig change
                sim = cosine_sim(old_partext[prevParIndex],new_partext[i])#compute topic similarity (tfidf)
                if sim<0.75:
                    sigChangePars.append(self.pars[self.latestVersion][i]) #significant change in topic similarity
                elif sim!=1:
                    smallChangePars.append(self.pars[self.latestVersion][i]) #small change
            else:
                self.addPars(None, i) #new node will be added to MIP
                addedPars.append(self.pars[self.latestVersion][i])
        
#        print 'old_new_mappings'
#        print old_new_mappings
#        print 'length'
#        print len(self.currentVersion.paragraphs)
        for i in range(0,len(self.currentVersion.paragraphs)):
            if old_new_mappings[i] is None:
                deletedPars.append(self.pars[self.latestVersion-1][i]) 
                self.mip.node[self.pars[self.latestVersion-1][i]]['deleted']=1
                
         
        #update user-paragraph edges weights for all relevant paragraphs        
        for par in deletedPars:
            self.updateEdge(userId, par, 'u-p', self.sigIncrement)
        for par in addedPars:
            self.updateEdge(userId, par, 'u-p', self.sigIncrement)
        for par in sigChangePars:
            self.updateEdge(userId, par, 'u-p', self.sigIncrement)
        for par in smallChangePars:
            self.updateEdge(userId, par, 'u-p', self.minIncrement)
            
        #update paragraph-paragraph edges weights
        bigChanges = addedPars+deletedPars+sigChangePars
        for i in range(0,len(bigChanges)):
            for j in range(i+1,len(bigChanges)):
                self.updateEdge(bigChanges[i], bigChanges[j], 'p-p', self.sigIncrement)
            for k in range(0,len(smallChangePars)):
                self.updateEdge(bigChanges[i], smallChangePars[k], 'p-p', self.minIncrement)
                
#        print 'added'
#        print addedPars
#        print 'deleted'
#        print deletedPars
#        print 'sigChanges'
#        print sigChangePars
#        print 'smallChange'
#        print smallChangePars
                    
#        print self.mip.nodes(True)
        
     
        
        #decay weights
#        for edge in self.mip.edges_iter(data=True):
#            if edge[2]['updated']==0:
#                if edge[2]['type']=='p-p':
#                    edge[2]['weight']=max(0,edge[2]['weight']-self.decay)
#                elif (edge[2]['type']=='u-p') & ((userId==edge[0]) | (userId==edge[0])):
#                    edge[2]['weight']=max(0,edge[2]['weight']-self.decay)
#                else:
#                    print 'not decaying'
                    
            
            
        self.currentVersion=newVersion
#        print'updating'
        
    def addUser(self,user_name):
        if (user_name in self.users):
            return self.users[user_name]
        else:
            self.lastID=self.lastID+1
            self.users[user_name] = self.lastID
            attr = {}
            attr['type']='user'
            self.mip.add_node(self.lastID, attr)
            self.nodeIdsToUsers[self.lastID]=user_name
        return self.users[user_name]
            
    def addPars(self,parPrevIndex,parNewIndex):
        if (parPrevIndex is not None):
            nodeId=self.pars[self.latestVersion-1][parPrevIndex]
            self.pars[self.latestVersion][parNewIndex]=nodeId
            self.mip.node[nodeId][self.latestVersion]=parNewIndex
#            print 'existing node'
#            print self.mip.node[nodeId]
        else:
            self.lastID=self.lastID+1
            self.pars[self.latestVersion][parNewIndex] = self.lastID
            attr = {}
            attr['type']='par'
            attr['deleted']=0
            attr[self.latestVersion]=parNewIndex
            self.mip.add_node(self.lastID, attr)
            
#            print 'new node'
#            print self.mip.node[self.lastID]
        return self.pars[self.latestVersion][parNewIndex]
            
    def updateEdge(self,i1,i2,type,increment = 1):
        if self.mip.has_edge(i1, i2):
            self.mip[i1][i2]['weight']=self.mip[i1][i2]['weight']+increment
        else:
            attr = {}
            attr['type']=type
            attr['weight']=increment
            self.mip.add_edge(i1, i2, attr)
        self.mip[i1][i2]['updated']=1


#class article:
#    def __init__(self, file_name):
#        self.current_pickle = wikiparser.get_pickle(file_name)
#        self.current_texts = self.current_pickle.get_all_text()
#        self.current_paras = self.current_pickle.get_all_paragraphs()
#        self.current_paratexts = [[a.text.encode('utf-8') for a in b] for b in self.current_paras]
#        self.current_names = self.current_pickle.get_all_authors()
#        

def get_pickle(pickle_file, folder="pickles"):
    pickle_name = os.path.join(os.getcwd(), folder, pickle_file)
    pkl_file = open(pickle_name, 'rb')
    print pkl_file
 #   try:
    xml_parse = cPickle.load(pkl_file)
    pkl_file.close()
  #  except:
#        print "Unexpected error:", sys.exc_info()[0]
#        print 'exception'
    return xml_parse 

def lev_sim(a, b):
    return Levenshtein.ratio(a, b)

def cosine_sim(a, b):
    if ((len(a)<5) | (len(b)<5)):
        return 1
    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix_train1 = tfidf_vectorizer.fit_transform((a, b))
    return cosine_similarity(tfidf_matrix_train1[0:1], tfidf_matrix_train1)[0][1]

def generate_ratios(paras1, paras2, reverse=False, sim_func=lev_sim):
    '''
    generates a list of lev distances between two lists of paragraphs
    return: list of list of lev ratios
    '''         
    dists = []
    if reverse:
        for a in paras2:
            lev_for_a = []
            for b in paras1:
                lev_for_a.append(sim_func(a, b))
            dists.append(lev_for_a)
    else:
        for a in paras1:
            lev_for_a = []
            for b in paras2:
                lev_for_a.append(sim_func(a, b))
            dists.append(lev_for_a)
    return dists

def generate_all_ratios(how=lev_sim):
    max_ratio_list = []
    for i in xrange(len(current_paras)):
        if i < len(current_paras)-1:
            dists_list = generate_ratios(current_paratexts[i], current_paratexts[i+1], reverse=False, sim_func=how)
            for a in dists_list:
                max_ratio_list.append( max(a))
    return max_ratio_list

store_pickle = True

#generates mapping between two paragraphs based on the lev dist between every combination of paragraph mappings
def assign_neighbors(version1, version2, delthreshhold):
    lev_dists = generate_ratios(version1, version2, reverse=False)
    mappings = {}
    for index1 in xrange(len(lev_dists)):
        index2, val = max([(i,x) for (i,x) in enumerate(lev_dists[index1])],key=lambda a:a[1])
        # if old paragraph maps to no new paragraphs, make note that it shouldnt be used
        if val < delthreshhold:
            mappings[index1] = None
        # if old paragraph maps well to a new paragraph, make note not to use anymore
        else:
            mappings[index1] = index2
    return mappings

def generate_mapping(mf,mb):
    new_mappings = {}
    found = False 
    for (key,value) in mf.iteritems():
        if value != None:
            if mb[value] == key:
                new_mappings[key] = value
                found = True
            else:
                new_mappings[key]=None
    mappings_reverse_direction= {}
    for (key,value) in new_mappings.iteritems():
        if value !=None: 
            mappings_reverse_direction[value]=key
        
            
    #return new_mappings
    return mappings_reverse_direction

def generate_mapping_old_new(mf,mb):
    new_mappings = {}
    found = False 
    for (key,value) in mf.iteritems():
        if value != None:
            if mb[value] == key:
                new_mappings[key] = value
                found = True
            else:
                new_mappings[key]=None
        else:
            new_mappings[key]=None

    #return new_mappings
    return new_mappings

def generate_mapping_for_revision(v1,v2):
    v1_paratext = [a.text.encode('utf-8') for a in v1.paragraphs]
    v2_paratext = [a.text.encode('utf-8') for a in v2.paragraphs]
    m_f = assign_neighbors(v1_paratext, v2_paratext, .4)
    m_b = assign_neighbors(v2_paratext, v1_paratext, .4)
    return (generate_mapping(m_f,m_b),generate_mapping_old_new(m_f,m_b))  

#parsing dropbox files

'''
-----------------------------------------------------------------------------
MIPs reasoning functions start
-----------------------------------------------------------------------------
'''
def DegreeOfInterestMIPs(mip, user, obj, alpha=0.3, beta=0.7):
    #compute apriori importance of node obj (considers effective conductance)
    current_flow_betweeness = nx.current_flow_betweenness_centrality(mip, True, 'weight');
    api_obj = current_flow_betweeness[obj]  #node centrality
#    print 'obj'
#    print obj
#    print 'api_obj'
#    print api_obj
    #compute proximity between user node and object node using Cycle-Free-Edge-Conductance from Koren et al. 2007
    cfec_user_obj = CFEC(user,obj,mip)
#    print 'cfec_user_obj'
#    print cfec_user_obj
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

'''
-----------------------------------------------------------------------------
MIPs reasoning functions end
-----------------------------------------------------------------------------
'''
 

if __name__ == '__main__':
    print 'test'
    #load necessary data
#    pickle_file_name = 'Absolute_pitch.pkl'
    pickle_file_name = 'Yale_University.pkl'
    current_pickle = get_pickle(pickle_file_name)
    print current_pickle
    current_texts = current_pickle.get_all_text()
    current_paras = current_pickle.get_all_paragraphs()
    print current_paras[0][0].text
    current_paratexts = [[a.text.encode('utf-8') for a in b] for b in current_paras]
    revision = current_pickle.revisions
    print len(current_paratexts[0])

    mip = Mip(revision[0])
    mip.initializeMIP()
    print len(revision)
    for i in range(0,len(revision)-1):
        print revision[i].author
        mip.updateMIP(revision[i])

    edgewidth=[]
    for (u,v,d) in mip.mip.edges(data=True):
        edgewidth.append(d['weight'])

#    elarge=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] >0.5]
#    esmall=[(u,v) for (u,v,d) in G.edges(data=True) if d['weight'] <=0.5]
    userNodes = [n for (n,d) in mip.mip.nodes(True) if d['type']=='user']
    parDeletedNodes = []
    parNodes = []
    for (n,d) in mip.mip.nodes(True):
        if d['type']=='par':
            if d['deleted']==1:
                parDeletedNodes.append(n)
            else:
                parNodes.append(n)

    pos=nx.spring_layout(mip.mip)
    new_labels = {}
    for node,d in mip.mip.nodes(True):
#        print 'node'
#        print node
#        print 'd'
#        print d
        if d['type']=='user':
            new_labels[node]=mip.nodeIdsToUsers[node]
#            print 'user'
        else:
            new_labels[node]=node
#            print 'par'
    
    
    
    
#    print DegreeOfInterestMIPs(mip.mip, 3, 7)
    nx.draw_networkx_nodes(mip.mip,pos,nodelist=userNodes,node_size=300,node_color='red')
    nx.draw_networkx_nodes(mip.mip,pos,nodelist=parNodes,node_size=300,node_color='blue')
    nx.draw_networkx_nodes(mip.mip,pos,nodelist=parDeletedNodes, node_size=300,node_color='black')
    nx.draw_networkx_edges(mip.mip,pos,edgelist=mip.mip.edges(),width=edgewidth)
    nx.draw_networkx_labels(mip.mip,pos,new_labels)
    print 'clustering'
    print(nx.average_clustering(mip.mip, weight = "weight"))
    #    G=nx.dodecahedral_graph()
#    nx.draw(mip.mip)
    plt.draw()
#    plt.savefig('ego_graph50.png')
    plt.show()
    