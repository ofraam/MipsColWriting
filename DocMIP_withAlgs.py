'''
Created on Jan 25, 2015

@author: Ofra
'''
import glob
import version 
#from version import Version
#from version import Page
import networkx as nx
import sys 
import cPickle, copy,  difflib, gensim, jellyfish
import Levenshtein, nltk, nltk.data, numpy as np, os, re
from difflib import SequenceMatcher
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import random
from nltk import recall
from nltk import precision

from sklearn.metrics import precision_recall_curve
import csv
import traceback
import operator

stop = stopwords.words('english')

class Session:
    def __init__(self, user, revision, time):
        self.actions = []
        self.user = user 
        self.time = time
        
    def __str__(self):
        toPrint = "session at time "+str(self.time) +" user = "+str(self.user)+"\n"
        for act in self.actions:
            toPrint = toPrint+str(act)+"\n"
        return toPrint
        
class Action:
    def __init__(self, user, ao, actType, desc, weightInc, changeExtent):
        self.user = user
        self.ao = ao
        self.actType = actType #view, edit, add, delete
        self.desc = desc
        self.weightInc = weightInc
        self.mipNodeID = -1
        self.changeExtent = changeExtent
        
        
    def updateMipNodeID(self, id):
        self.mipNodeID = id
        
    def __str__(self):
        return "user = "+str(self.user) +"\n"+"ao = "+str(self.ao) +"\n" + "actType = "+self.actType +"\n" + "desc = "+self.desc +"\n" + "weightInc = "+str(self.weightInc) +"\n" + "extent of change = "+str(self.changeExtent) +"\n" + "mipNodeID = "+str(self.mipNodeID) +"\n"
        

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
        self.current_flow_betweeness = None #centrality values of all nodes
        self.log = [] #log holds all the session data 
        self.anonCount = 0 #counts how many anonymous edits were made (so they do not collapse to the same node because they have the "same" user name)
        

    def initializeMIP(self):
        self.pars.append({})
        userId = self.addUser(self.currentVersion.author)
        
        session = Session(self.currentVersion.author, self.currentVersion, len(self.log))
        partext = [a.text.encode('utf-8') for a in self.currentVersion.paragraphs]
#        print userId
        index = 0
        for par in self.currentVersion.paragraphs:
            parId = self.addPars(None, index)
#            print parId
            self.updateEdge(userId, parId,'u-p', self.sigIncrement)
            session.actions.append(Action(session.user, parId, 'sigEdit',partext[index], self.sigIncrement,1))
            index=index+1
            
        for i in range(0,len(self.currentVersion.paragraphs)):
            for j in range(i+1,len(self.currentVersion.paragraphs)):
                if i!=j:
                    self.updateEdge(self.pars[self.latestVersion-1][i], self.pars[self.latestVersion][j],'p-p', self.sigIncrement)
                    
        self.log.append(session)
        self.current_flow_betweeness = nx.current_flow_betweenness_centrality(self.mip,True, 'weight')
#        print len(self.log)
#        print self.mip.nodes(True)
        
        
                
    def updateMIP(self, newVersion):
        #create session object
        session = Session(newVersion.author, newVersion, len(self.log))
        
        
        self.pars.append({})
        self.latestVersion=self.latestVersion+1
        #clear updated edges
        for edge in self.mip.edges_iter(data=True):
            edge[2]['updated']=0
            
        #get user
        userId = self.addUser(newVersion.author)
            
      
        (new_old_mappings,old_new_mappings) = generate_mapping_for_revision(self.currentVersion,newVersion)
        mappings=new_old_mappings


        #check for significant changes, additions and deletions; add to MIP
        old_partext = [a.text.encode('utf-8') for a in self.currentVersion.paragraphs]
        new_partext = [a.text.encode('utf-8') for a in newVersion.paragraphs]
        sigChangePars=[]
        smallChangePars=[]
        addedPars=[]
        deletedPars=[] #note this is from *previous* revision

        for i in range(0,len(newVersion.paragraphs)):
            if i in mappings: #node in MIP already exists, just update
#                print mappings[i]
                prevParIndex=mappings[i]
                self.addPars(prevParIndex, i) # adding to MIP
                #check for sig change
#                print 'old_partext[prevParIndex] '
##                print old_partext[prevParIndex]
#                print len(old_partext[prevParIndex])
##                print 'new_partext[i]'
#                print new_partext[i]
#                print len(new_partext[i])
                sim = cosine_sim(old_partext[prevParIndex],new_partext[i])#compute topic similarity (tfidf)
                if sim<0.75:
                    sigChangePars.append(self.pars[self.latestVersion][i]) #significant change in topic similarity
                    session.actions.append(Action(session.user, self.pars[self.latestVersion][i], 'sigEdit',new_partext[i], self.sigIncrement, 1-sim))
                elif sim!=1: #make sure par changed at all
                    smallChangePars.append(self.pars[self.latestVersion][i]) #small change
                    session.actions.append(Action(session.user, self.pars[self.latestVersion][i], 'smallEdit',new_partext[i], self.minIncrement, 1-sim))
            else:
                self.addPars(None, i) #new node will be added to MIP
                addedPars.append(self.pars[self.latestVersion][i])
                session.actions.append(Action(session.user, self.pars[self.latestVersion][i], 'added',new_partext[i], self.sigIncrement, 1)) #added new paragraph, so extent of change is 1 out of 1 
                
        

        for i in range(0,len(self.currentVersion.paragraphs)):
            if old_new_mappings[i] is None:
                deletedPars.append(self.pars[self.latestVersion-1][i]) 
                self.mip.node[self.pars[self.latestVersion-1][i]]['deleted']=1
                session.actions.append(Action(session.user, self.pars[self.latestVersion-1][i], 'deleted',"", self.sigIncrement, 1)) #deleted paragraph, so extent of change is 1 out of 1
                
         
        #update user-paragraph edges weights for all relevant paragraphs        
        for par in deletedPars:
            self.updateEdge(userId, par, 'u-p', self.sigIncrement)
        for par in addedPars:
            self.mip.node[par]['allEditRevs'].append(self.latestVersion)
            self.mip.node[par]['sigEditRevs'].append(self.latestVersion)
            self.updateEdge(userId, par, 'u-p', self.sigIncrement)
        for par in sigChangePars:
            self.mip.node[par]['allEditRevs'].append(self.latestVersion) #TODO: check fix is correct
            self.mip.node[par]['sigEditRevs'].append(self.latestVersion)
            self.updateEdge(userId, par, 'u-p', self.sigIncrement)
        for par in smallChangePars:
            self.mip.node[par]['allEditRevs'].append(self.latestVersion)
            self.updateEdge(userId, par, 'u-p', self.minIncrement)
            
        #update paragraph-paragraph edges weights
        bigChanges = addedPars+deletedPars+sigChangePars
        for i in range(0,len(bigChanges)):
            for j in range(i+1,len(bigChanges)):
                self.updateEdge(bigChanges[i], bigChanges[j], 'p-p', self.sigIncrement)
            for k in range(0,len(smallChangePars)):
                self.updateEdge(bigChanges[i], smallChangePars[k], 'p-p', self.minIncrement)
  
        
        #decay weights TODO: consider only decaying relationships that had an opportunity to be observed.
#        for edge in self.mip.edges_iter(data=True):
#            if edge[2]['updated']==0:
#                if edge[2]['type']=='p-p':
#                    edge[2]['weight']=max(0,edge[2]['weight']-self.decay)
#                elif (edge[2]['type']=='u-p') & ((userId==edge[0]) | (userId==edge[0])):
#                    edge[2]['weight']=max(0,edge[2]['weight']-self.decay)
#                else:
#                    print 'not decaying'
                    
            
        #update current version and log with new session
        self.currentVersion=newVersion
        self.log.append(session)
        try:
#            self.current_flow_betweeness = nx.current_flow_betweenness_centrality(self.mip,True, weight = 'weight')
            self.current_flow_betweeness = nx.degree_centrality(self.mip)
        except:
            self.current_flow_betweeness = nx.degree_centrality(self.mip)
#        self.current_flow_betweeness = nx.betweenness_centrality(self.mip,True, weight = 'weight')

#        print 'session'
#        print len(session.actions)
#        print "session user = "+session.user
#        print'updating'
        
    def addUser(self,user_name):
#        print user_name
        if (user_name in self.users):
            self.mip.node[self.users[user_name]]['numRevisions'] = self.mip.node[self.users[user_name]]['numRevisions']+1 #update the number of revisions by this user
            return self.users[user_name]
        else:
            self.lastID=self.lastID+1
           
            if user_name=="":
                user_name = "anon_"+str(self.anonCount)
                self.anonCount = self.anonCount+1
            self.users[user_name] = self.lastID    
            attr = {}
            attr['type']='user'
            attr['numRevisions']=1
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
            attr['sigEditRevs']=[]
            attr['allEditRevs']=[]
            attr[self.latestVersion]=parNewIndex
            self.mip.add_node(self.lastID, attr)
            
#            print 'new node'
#            print self.mip.node[self.lastID]
        return self.pars[self.latestVersion][parNewIndex]
            
    def updateEdge(self,i1,i2,typeEdit,increment = 1):
        if self.mip.has_edge(i1, i2):
            self.mip[i1][i2]['weight']=self.mip[i1][i2]['weight']+increment
        else:
            attr = {}
            attr['type']=typeEdit
            attr['weight']=increment
            self.mip.add_edge(i1, i2, attr)
        self.mip[i1][i2]['updated']=1
        
    def getLiveObjects(self):
        liveObjects = []
        for node in self.mip.nodes(True):
            if node[1]['type']=='par':
                if node[1]['deleted']==0:
                    liveObjects.append(node)
        return liveObjects
    
    def getLiveAos(self):
        liveObjects = []
        for node in self.mip.nodes(True):
            if node[1]['type']=='par':
                if node[1]['deleted']==0:
                    liveObjects.append(node[0])
        return liveObjects
    '''
    -----------------------------------------------------------------------------
    MIPs reasoning functions start
    -----------------------------------------------------------------------------
    '''
    def DegreeOfInterestMIPs(self, user, obj, current_flow_betweeness, alpha=0.3, beta=0.7, similarity = "adamic"):
        #compute apriori importance of node obj (considers effective conductance)
#        current_flow_betweeness = nx.current_flow_betweenness_centrality(self.mip,True, 'weight');
        
        api_obj = current_flow_betweeness[obj]  #node centrality
    #    print 'obj'
    #    print obj
    #    print 'api_obj'
    #    print api_obj
        #compute proximity between user node and object node using Cycle-Free-Edge-Conductance from Koren et al. 2007
        proximity = 0
        if ((user in self.users) & (beta>0)): #no point to compute proximity if beta is 0... (no weight)
            userID = self.users[user]
            if similarity == "adamic":
                proximity = self.adamicAdarProximity(userID,obj) #Adamic/Adar proximity
#                print 'computing proximity'
            else:
#                proximity = self.CFEC(userID,obj) #cfec proximity
                proximity = self.edgeBasedProximity(userID, obj, 0.7)
        else:
            return alpha*api_obj
#        print 'api_obj = '+str(api_obj)
#        print 'proximity = '+str(proximity)
        return alpha*api_obj+beta*proximity #TODO: check that scales work out for centrality and proximity, otherwise need some normalization


    '''
    computes Adamic/Adar proximity between nodes, adjusted to consider edge weights
    here's adamic/adar implementation in networkx. Modifying to consider edge weights            
    def predict(u, v):
        return sum(1 / math.log(G.degree(w))
                   for w in nx.common_neighbors(G, u, v))
    '''
    def simpleProximity(self, s, t): #s and t are the mip node IDs, NOT user/obj ids
        proximity = 0.0
        sharedWeight = 0.0
        for node in nx.common_neighbors(self.mip, s, t):
            sharedWeight = sharedWeight + self.mip[s][node]['weight'] + self.mip[t][node]['weight'] #the weight of the path connecting s and t through the current node
        proximity = sharedWeight/(self.mip.degree(s, weight = 'weight')+self.mip.degree(t, weight = 'weight')+0.000000000001)
        return proximity  
    
    
    def edgeBasedProximity(self,s,t, edgeWeight=0.7):
#        print 'correct function'
        simpleProximity = self.simpleProximity(s, t)
        edgeProximity=0.0
        if self.mip.has_edge(s, t):
            edgeProximity = self.mip[s][t]['weight']/self.mip.degree(s, weight = 'weight')
#            print 'there is an edge'
        return edgeWeight*edgeProximity+(1-edgeWeight)*simpleProximity 

    def adamicAdarProximity(self, s, t):
        proximity = 0.0
        for node in nx.common_neighbors(self.mip, s, t):
            weights = self.mip[s][node]['weight'] + self.mip[t][node]['weight'] #the weight of the path connecting s and t through the current node
            if weights!=0: #0 essentially means no connection
#                print 'weights = '+str(weights)
#                print 'degree = '+str(self.mip.degree(node, weight = 'weight'))
                proximity = proximity + (weights*(1/(math.log(self.mip.degree(node, weight = 'weight'))+0.00000000000000000000000001))) #gives more weight to "rare" shared neighbors, adding small number to avoid dividing by zero
#                print 'proximity = '+str(proximity)
        return proximity    
    '''
    computes Cycle-Free-Edge-Conductance from Koren et al. 2007
    for each simple path, we compute the path probability (based on weights) 
    '''
    def CFEC(self,s,t):
        R = nx.all_simple_paths(self.mip, s, t, cutoff=3)
        proximity = 0.0
        for r in R:
            PathWeight = self.mip.degree(r[0])*(self.PathProb(r))  #check whether the degree makes a difference, or is it the same for all paths??
            proximity = proximity + PathWeight
            
            
        return proximity
        
            
    def PathProb(self, path):
        prob = 1.0
        for i in range(len(path)-1):
            prob = prob*(float(self.mip[path[i]][path[i+1]]['weight'])/self.mip.degree(path[i]))
#        print 'prob' + str(prob)
        return prob
    
    '''
    rank all live objects based on DOI to predict what edits a user will make.
    NOTE: need to call this function with the mip prior to the users' edits!!!
    '''
    def rankLiveObjectsForUser(self, user, alpha = 0.3, beta = 0.7, similarity = "adamic"):
        aoList = self.getLiveAos()
#        print 'number of pars = '+str(len(aoList))
        notificationsList = []
        for ao in aoList:
            doi = self.DegreeOfInterestMIPs(user, ao,self.current_flow_betweeness, alpha, beta, similarity)  
            
            if len(notificationsList)==0:
                toAdd = []
                toAdd.append(ao)
                toAdd.append(doi)
                notificationsList.append(toAdd)
            else:
                j = 0
                while ((doi<notificationsList[j][1])):
                    if j<len(notificationsList)-1:
                        j = j+1
                    else:
                        j=j+1
                        break
                toAdd = []
                toAdd.append(ao)
                toAdd.append(doi)                  
                if (j<len(notificationsList)):
                    notificationsList.insert(j, toAdd)
                else:
                    notificationsList.append(toAdd)  
#        print 'notification list size = '+str(len(notificationsList))        
        return notificationsList
    
    def rankChangesForUser(self,user,time, onlySig = True, alpha = 0.3, beta = 0.7, similarity = "adamic"):
        notificationsList = []
        checkedObjects = {}
        for i in range(time, len(self.log)-1): #this includes revision at time TIME and does not include last revision in MIP, which is the one when the user is back 
#            print "time = "+str(i) + "author = "+self.log[i].user
                        
            session = self.log[i]
            for act in session.actions: 

                if ((act.actType != 'smallEdit') | (onlySig == False)):
                    if (act.ao not in checkedObjects): #currently not giving more weight to the fact that an object was changed multiple times. --> removed because if there are both big and small changes etc...
                        #TODO: possibly add check whether the action is notifiable
                        
                        doi = self.DegreeOfInterestMIPs(user, act.ao,self.current_flow_betweeness, alpha, beta, similarity)
                        checkedObjects[act.ao] = doi
#                    else:
#                        doi = checkedObjects[act.ao] #already computed doi, don't recompute!
                        #put in appropriate place in list based on doi
                        if len(notificationsList)==0:
                            toAdd = []
                            toAdd.append(act)
                            toAdd.append(doi)
                            notificationsList.append(toAdd)
                        else:
                            j = 0
    
                            while ((doi<notificationsList[j][1])):
                                if j<len(notificationsList)-1:
                                    j = j+1
                                else:
                                    j=j+1
                                    break
                            toAdd = []
                            toAdd.append(act)
                            toAdd.append(doi)   
                         
                            if (j<len(notificationsList)):
                                notificationsList.insert(j, toAdd)
                            else:
                                notificationsList.append(toAdd)                        
        return notificationsList
    
    def rankChangesGivenUserFocus(self,user,focus_obj, time):
        notificationsList = []
        checkedObjects = []
        for i in range(time, len(self.log)-1):
            session = self.log[i]
            for act in session.actions:
                if (act.ao not in checkedObjects):
                    #TODO: possibly add check whether the action is notifiable
                    doi = self.DegreeOfInterestMIPs(focus_obj, act.ao)
                    #put in appropriate place in list based on doi
                    if (len(notificationsList==0)):
                        notificationsList.append(act.ao, doi)
                    else:
                        j = 0
                        while (doi<notificationsList[j][1]):
                            j = j+1
                        notificationsList.insert(j, act.ao)
                        
        return notificationsList

#    def rankChangeImplications(self,target_obj, time):
#        notificationsList = []
#        checkedObjects = []
#        for i in range(time, len(self.log)):
#            session = self.log[i]
#            for act in session.actions:
#                if (act.ao not in checkedObjects):
#                    #TODO: possibly add check whether the action is notifiable
#                    doi = self.DegreeOfInterestMIPs(self.objects[target_obj], self.objects[act.ao])
#                    #put in appropriate place in list based on doi
#                    if (len(notificationsList==0)):
#                        notificationsList.append(act.ao, doi)
#                    else:
#                        j = 0
#                        while (doi<notificationsList[j][1]):
#                            j = j+1
#                        notificationsList.insert(j, act.ao)
#                        
#        return notificationsList                    
                
    '''
    -----------------------------------------------------------------------------
    MIPs reasoning functions end
    -----------------------------------------------------------------------------
    '''

'''
-----------------------------------------------------------------------------
Writing funcs
----------------------------------------------------------------------------
'''
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

#def generate_all_ratios(how=lev_sim):
#    max_ratio_list = []
#    for i in xrange(len(current_paras)):
#        if i < len(current_paras)-1:
#            dists_list = generate_ratios(current_paratexts[i], current_paratexts[i+1], reverse=False, sim_func=how)
#            for a in dists_list:
#                max_ratio_list.append( max(a))
#    return max_ratio_list
#
#store_pickle = True

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
Writing funcs end
----------------------------------------------------------------------------
'''

'''
-----------------------------------------------------------------------------
eval funcs
----------------------------------------------------------------------------
'''

def evaluateGeneralParagraphRankingForAuthors(articleRevisions, articleName):
    
    last_author_revs = {}
    author_rev_counts = {}
    author_rev_counts[''] = 1 #always true for anonymous...
    prev_author = None
    prediction_type = '' #returning user, first time user or anonymous user?
#    last_author_revs[articleRevisions.revisions[0].author] = 0 #initialize with first author
#    for i in range(len(articleRevisions.revisions)):
    results = []
    for i in range(2,len(articleRevisions.revisions)):
        j=0
        print i
        mip = readMIPfromFile(articleName,i-1) #taking the mip up until the last author, so we can look at all current paragraph and rank them for the user. 
        cur_author = articleRevisions.revisions[i].author
#        if (cur_author == prev_author):
#            print 'strange'
        if ((cur_author in last_author_revs) & (cur_author!="")) : #check that there was a previous revision for this author (otherwise can't compute DOI, though could do just api, maybe add later), and that the author is not anonymous (have no info)
            if last_author_revs[cur_author]<i-1: #if the author wrote the previous revision, nothing to do.
                j = 1;
                if (i+j<len(articleRevisions.revisions)):
                    while ((articleRevisions.revisions[i+j].author==cur_author) & (i+j<len(articleRevisions.revisions) - 1)):
                        j = j+1
                
                
                rankings = {}
                
                rankings["doi_alpha1_beta0"] = mip.rankLiveObjectsForUser(cur_author,alpha = 1.0, beta = 0.0,similarity="edge")
                
                rankings["doi_alpha0_beta1"] = mip.rankLiveObjectsForUser(cur_author,alpha = 0.0, beta = 1.0,similarity="edge")
                rankings["doi_alpha05_beta05"] = mip.rankLiveObjectsForUser(cur_author,alpha = 0.5, beta = 0.5,similarity="edge")
                rankings["doi_alpha03_beta07"] = mip.rankLiveObjectsForUser(cur_author,alpha = 0.5, beta = 0.5,similarity="edge")
                listOfPars =  [ row for row in rankings["doi_alpha03_beta07"] ] 
                
                #generate baselines rankings: random, top edit, recently edited (TODO: language model):                
                rankings['random'] = generateRandomRanking(listOfPars) #random
                rankings['recency'] = generateRankingByRecency(listOfPars, mip, sigEdit = True) #recency, currently only sig edits
                rankings['topEdited10'] = generateRankingByTopEdited(listOfPars, mip, revWindow = 10, sigEdit = True) #top edits, currently only sig edits and in the past 10 revisions
                rankings['topEdited200'] = generateRankingByTopEdited(listOfPars, mip, revWindow = 200, sigEdit = True)
                last_author_revs[cur_author] = i+j-1 #update latest revision made by this author
                author_rev_counts[cur_author] = author_rev_counts[cur_author]+1
                prediction_type = 'returning'
            else: #should be unnecessary 
                prev_author = cur_author
        else: #either anonymous author, or first time author. In these cases no point of doing doi, but can still do api
            if cur_author == "": #anonymous
                if i>0:
                    j=1
                    rankings = {}
                    rankings["doi_alpha1_beta0"] = mip.rankLiveObjectsForUser(cur_author,alpha = 1.0, beta = 0.0,similarity="edge")
                    listOfPars =  [ row for row in rankings["doi_alpha1_beta0"] ] 
                  
                    #generate baselines rankings: random, top edit, recently edited (TODO: language model):                
                    rankings['random'] = generateRandomRanking(listOfPars) #random
                    rankings['recency'] = generateRankingByRecency(listOfPars, mip, sigEdit = True) #recency, currently only sig edits
                    rankings['topEdited10'] = generateRankingByTopEdited(listOfPars, mip, revWindow = 10, sigEdit = True) #top edits, currently only sig edits and in the past 10 revisions  
                    rankings['topEdited200'] = generateRankingByTopEdited(listOfPars, mip, revWindow = 200, sigEdit = True)
                    
                    prediction_type = 'anonymous'               
                    #TODO: random ranking, recent ranking, size of change ranking
            else: #not anonymous, but first time user
                j = 1

                if (i+j<len(articleRevisions.revisions)):
                    while ((articleRevisions.revisions[i+j].author==cur_author) & (i+j<len(articleRevisions.revisions)-1)):
                        j = j+1
                if i>0:
                    rankings = {}
                    rankings["doi_alpha1_beta0"] = mip.rankLiveObjectsForUser(cur_author,alpha = 1.0, beta = 0.0,similarity="edge")
                    listOfPars =  [ row for row in rankings["doi_alpha1_beta0"] ] 
                    #generate baselines rankings: random, top edit, recently edited (TODO: language model):                
                    rankings['random'] = generateRandomRanking(listOfPars) #random
                    rankings['recency'] = generateRankingByRecency(listOfPars, mip, sigEdit = True) #recency, currently only sig edits
                    rankings['topEdited10'] = generateRankingByTopEdited(listOfPars, mip, revWindow = 10, sigEdit = True) #top edits, currently only sig edits and in the past 10 revisions
                    rankings['topEdited200'] = generateRankingByTopEdited(listOfPars, mip, revWindow = 200, sigEdit = True)
                    prediction_type = 'firstTime'                 
                    #TODO: random ranking, recent ranking, size of change ranking
                last_author_revs[cur_author] = i+j-1 #update latest revision made by this author
                author_rev_counts[cur_author] = 1
        #get actual edits of the author in their current revision(s)
        
        if ((i>0) & ((prev_author!=cur_author) | (cur_author == ""))):
            actualEditsFull = getSignificantEditsOfAuthor(cur_author, i, i+j, articleName) #TODO: check i+j is right, or need i+j+1
            actualEdits = generateChangeListWithObjectIDs(actualEditsFull)
            
           
            livePars = mip.getLiveObjects();
            
            #evaluate all rankings        
            if len(actualEdits)>0:
                for rank in rankings:
                    changeListFull =  [ row[0] for row in rankings[rank] ] 
#                    print 'changeListFull: '+str(changeListFull)
                    changedObjects = generateChangeListWithObjectIDsJustAO(changeListFull)
#                    print 'change list set: '+str(changedObjects)

                    cur_authorForPrint= cur_author.encode('utf-8')
                    for k in range(1,len(changedObjects)+1): #TODO check makes sense
                        resultsForRanking = []
                        resultsForRanking.append(rank) #add name of ranker for writing to file later
                        resultsForRanking.append(i)                    
                        resultsForRanking.append(cur_authorForPrint)
                        resultsForRanking.append(author_rev_counts[cur_author])
                        resultsForRanking.append(prediction_type)
                        resultsForRanking.append(len(actualEdits))
                        resultsForRanking.append(len(livePars))
                        resultsForRanking.append(len(changedObjects)) #how many pars were ranked by alg
                        resultsForRanking.append(getOverlap(actualEdits, changedObjects)) #how many pars were ranked & changed
                        resultsForRanking.append(str(k)+"_precision")                        
                        precision = precisionAtN(actualEdits, changedObjects,k)
                        resultsForRanking.append(precision)
                        results.append(resultsForRanking)
                        
                        resultsForRanking = []
                        resultsForRanking.append(rank) #add name of ranker for writing to file later
                        resultsForRanking.append(i)                    
                        resultsForRanking.append(cur_authorForPrint)
                        resultsForRanking.append(author_rev_counts[cur_author])
                        resultsForRanking.append(prediction_type)
                        resultsForRanking.append(len(actualEdits))
                        resultsForRanking.append(len(livePars))
                        resultsForRanking.append(len(changedObjects)) #how many pars were ranked by alg
                        resultsForRanking.append(getOverlap(actualEdits, changedObjects)) #how many pars were ranked & changed
                        resultsForRanking.append(str(k)+"_recall")                        
                        recall = recallAtN(actualEdits, changedObjects,k)
                        resultsForRanking.append(recall)
                        results.append(resultsForRanking)

#                        if recall ==1: #if we retrieved all changes made by user, no point to proceed (not implemented for now because different rankers may complete at different number)
#                            break;
                    
        prev_author = cur_author
    return results


'''
evaluates the performance of change ranking mechanisms in terms of precision and recall . 
Here, there's an assumption that if we inform an author that a paragraph changed we think the author should edit that paragraph
Current setting only looks at significant edits. Can change.
'''
def evaluateChangesForAuthors(articleRevisions, articleName):
    last_author_revs = {}
    author_rev_counts = {}
    author_rev_counts[''] = 1 #always true for anonymous...
    prev_author = None
    prediction_type = '' #returning user, first time user or anonymous user?
    
#    last_author_revs[articleRevisions.revisions[0].author] = 0 #initialize with first author
#    for i in range(len(articleRevisions.revisions)):
    results = []
    for i in range(len(articleRevisions.revisions)):
        j=0
#        print i
        mip = readMIPfromFile(articleName,i) #taking the mip including the current author, but the ranking changes function will do the right thing (rank until the previous session)
        cur_author = articleRevisions.revisions[i].author
#        if (cur_author == prev_author):
#            print 'strange'
        if ((cur_author in last_author_revs) & (cur_author!="")) : #check that there was a previous revision for this author (otherwise can't compute DOI, though could do just api, maybe add later), and that the author is not anonymous (have no info)
            if last_author_revs[cur_author]<i-1: #if the author wrote the previous revision, nothing to do.
                j = 1;
                if (i+j<len(articleRevisions.revisions)):
                    while ((articleRevisions.revisions[i+j].author==cur_author) & (i+j<len(articleRevisions.revisions)-1)):
                        j = j+1
                
                
                rankings = {}
                
                rankings["doi_alpha1_beta0"] = mip.rankChangesForUser(cur_author,last_author_revs[cur_author]+1,False,alpha = 1.0, beta = 0.0,similarity="edge")
                rankings["doi_alpha0_beta1"] = mip.rankChangesForUser(cur_author,last_author_revs[cur_author]+1,False, alpha = 0.0, beta = 1.0,similarity="edge")
                rankings["doi_alpha05_beta05"] = mip.rankChangesForUser(cur_author,last_author_revs[cur_author]+1,False, alpha = 0.5, beta = 0.5,similarity="edge")
                rankings["doi_alpha03_beta07"] = mip.rankChangesForUser(cur_author,last_author_revs[cur_author]+1,False, alpha = 0.3, beta = 0.7,similarity="edge")
                listOfChangedPars =  [ row for row in rankings["doi_alpha03_beta07"] ] 


            
                #generate baselines rankings: random, top edit, recently edited (TODO: language model):                
                rankings['random'] = generateRandomRanking(listOfChangedPars) #random
                rankings['recency'] = generateRankingByRecencyActionList(listOfChangedPars, mip, sigEdit = True) #recency, currently only sig edits
                rankings['topEdited10'] = generateRankingByTopEditedActionsList(listOfChangedPars, mip, revWindow = 10, sigEdit = True) #top edits, currently only sig edits and in the past 10 revisions
                rankings['topEdited200'] = generateRankingByTopEditedActionsList(listOfChangedPars, mip, revWindow = 200, sigEdit = True)
                last_author_revs[cur_author] = i+j-1 #update latest revision made by this author
                author_rev_counts[cur_author] = author_rev_counts[cur_author]+1
                prediction_type = 'returning'
            else: #should be unnecessary 
                prev_author = cur_author
        else: #either anonymous author, or first time author. In these cases no point of doing doi, but can still do api
            if cur_author == "": #anonymous
                if i>0:
                    j=1
                    rankings = {}
                    rankings["doi_alpha1_beta0"] = mip.rankChangesForUser(cur_author,max(0,i-10),False, alpha = 1.0, beta = 0.0,similarity="edge")
                    listOfChangedPars =  [ row for row in rankings["doi_alpha1_beta0"] ] 
                    #generate baselines rankings: random, top edit, recently edited (TODO: language model):                
                    rankings['random'] = generateRandomRanking(listOfChangedPars) #random
                    rankings['recency'] = generateRankingByRecencyActionList(listOfChangedPars, mip, sigEdit = True) #recency, currently only sig edits
                    rankings['topEdited10'] = generateRankingByTopEditedActionsList(listOfChangedPars, mip, revWindow = 10, sigEdit = True) #top edits, currently only sig edits and in the past 10 revisions   
                    rankings['topEdited200'] = generateRankingByTopEditedActionsList(listOfChangedPars, mip, revWindow = 200, sigEdit = True)                   
                    prediction_type = 'anonymous'                 
                    #TODO: random ranking, recent ranking, size of change ranking
            else: #not anonymous, but first time user
                j = 1

                if (i+j<len(articleRevisions.revisions)):
                    while ((articleRevisions.revisions[i+j].author==cur_author) & (i+j<len(articleRevisions.revisions)-1)):
                        j = j+1
                if i>0:
                    rankings = {}
                    rankings["doi_alpha1_beta0"] = mip.rankChangesForUser(cur_author,max(0,i-10),False, alpha = 1.0, beta = 0.0,similarity="edge")
                    listOfChangedPars =  [ row for row in rankings["doi_alpha1_beta0"] ] 
                   
                    #generate baselines rankings: random, top edit, recently edited (TODO: language model):                
                    rankings['random'] = generateRandomRanking(listOfChangedPars) #random
                    rankings['recency'] = generateRankingByRecencyActionList(listOfChangedPars, mip, sigEdit = True) #recency, currently only sig edits
                    rankings['topEdited10'] = generateRankingByTopEditedActionsList(listOfChangedPars, mip, revWindow = 10, sigEdit = True) #top edits, currently only sig edits and in the past 10 revisions
                    rankings['topEdited200'] = generateRankingByTopEditedActionsList(listOfChangedPars, mip, revWindow = 200, sigEdit = True)                  
                    prediction_type = 'firstTime'                   
                    #TODO: random ranking, recent ranking, size of change ranking
                last_author_revs[cur_author] = i+j-1 #update latest revision made by this author
                author_rev_counts[cur_author] = 1
        
        #get actual edits of the author in their current revision(s)

            
        
        if ((i>0) & ((prev_author!=cur_author) | (cur_author == ""))):
#            for r in rankings.items():
#                print len(r[1])
            actualEditsFull = getSignificantEditsOfAuthor(cur_author, i, i+j, articleName) #TODO: check i+j is right, or need i+j+1
            actualEdits = generateChangeListWithObjectIDs(actualEditsFull)

            mipBeforeRevision = readMIPfromFile(articleName,i-1) #the mip before current edit, see how many paragraphs were there
            livePars = mipBeforeRevision.getLiveObjects();
            
            
            
            if len(actualEdits) > len(livePars):
                print 'strange'
            #evaluate all rankings        
            if len(actualEdits)>0:
                for rank in rankings:
                    changeListFull =  [ row[0] for row in rankings[rank] ] 
                    if ((rank == 'recency') | (rank == 'topEdited10') | (rank == 'topEdited200')):
                        changedObjects = generateChangeListWithObjectIDsJustAO(changeListFull)
                    else:
                        changedObjects = generateChangeListWithObjectIDs(changeListFull)
                    #remove deleted paragraphs as they cannot be edited again and no point of predicting they will be edited... [will not be true when we recommend authors what to look at, however!]
                    for obj in changedObjects:
                        if obj not in livePars:
                            changedObjects.remove(obj)
                    cur_authorForPrint= cur_author.encode('utf-8')
                    for k in range(1,len(changedObjects)+1): #TODO check makes sense
                        resultsForRanking = []
                        resultsForRanking.append(rank) #add name of ranker for writing to file later
                        resultsForRanking.append(i)                    
                        resultsForRanking.append(cur_authorForPrint)
                        resultsForRanking.append(author_rev_counts[cur_author])
                        resultsForRanking.append(prediction_type)
                        resultsForRanking.append(len(actualEdits)) #how many pars author changed
                        resultsForRanking.append(len(livePars)) #how many pars were "live" before author revision
                        resultsForRanking.append(len(changedObjects)) #how many pars were ranked by alg
                        resultsForRanking.append(getOverlap(actualEdits, changedObjects)) #how many pars were ranked & changed
                        resultsForRanking.append(str(k)+"_precision")                        
                        precision = precisionAtN(actualEdits, changedObjects,k)
                        resultsForRanking.append(precision)
                        results.append(resultsForRanking)
                        
                        resultsForRanking = []
                        resultsForRanking.append(rank) #add name of ranker for writing to file later
                        resultsForRanking.append(i)                    
                        resultsForRanking.append(cur_authorForPrint)
                        resultsForRanking.append(author_rev_counts[cur_author])
                        resultsForRanking.append(prediction_type)
                        resultsForRanking.append(len(actualEdits))
                        resultsForRanking.append(len(livePars))
                        resultsForRanking.append(len(changedObjects)) #how many pars were ranked by alg
                        resultsForRanking.append(getOverlap(actualEdits, changedObjects)) #how many pars were ranked & changed
                        resultsForRanking.append(str(k)+"_recall")                        
                        recall = recallAtN(actualEdits, changedObjects,k)
                        resultsForRanking.append(recall)
                        results.append(resultsForRanking)
                        
        prev_author = cur_author
    return results

def getOverlap(actual, predicted):
    counter = 0;
    for a in actual:
        if a in predicted:
            counter = counter+1
    return counter

def checkIfnonActualInChanged(actual, predicted):
    for a in actual:
        if a in predicted:
            return True;
    return False;
                   

def getEditsOfAuthor(author, startRev, endRev, articleName):
    mip = readMIPfromFile(articleName,endRev-1) #want the mip up to previous rev
    changeList = []
#    print "len(mip.log)" + str(len(mip.log))
#    print "endRev = " + str(endRev)
    for i in range(startRev,endRev):#endRev should be the revision that has the NEXT author (not the last of current author)
        session = mip.log[i]
        for act in session.actions:
            changeList.append(act)            
    return changeList

def getSignificantEditsOfAuthor(author, startRev, endRev, articleName):
#    print 'getting edits of user: '+author + 'from '+str(startRev)+ ' to'+str(endRev)
    mip = readMIPfromFile(articleName,endRev-1) #want the mip up to previous rev
    changeList = []
#    print "len(mip.log)" + str(len(mip.log))
#    print "endRev = " + str(endRev)
    for i in range(startRev,endRev):#endRev should be the revision that has the NEXT author (not the last of current author)
        session = mip.log[i]
#        print 'author is correct: '+str(author == session.user)
        if (author!=session.user):
            print 'bug'
        for act in session.actions:
#            if ((act.actType != 'smallEdit') & (act.actType !='added')): #not looking at small edits, but also not on added paragraph as those cannot be predicted
            if (act.actType !='added'): #not looking at small edits, but also not on added paragraph as those cannot be predicted
                changeList.append(act)            
    return changeList

def writeResultsToFile(results, fileName):
    resultFile = open(fileName,'wb')
    wr = csv.writer(resultFile, dialect='excel')
    for ranker in results:
        wr.writerow(ranker)
          
def generateChangeListWithObjectIDs(changes): #one entry for each paragraph that change
    prunedList = []
    for change in changes:
        if change.ao not in prunedList:
            prunedList.append(change.ao)
    return prunedList

def generateChangeListWithObjectIDsJustAO(changes): #one entry for each paragraph that change
    prunedList = []
    for ao in changes:
        if ao not in prunedList:
            prunedList.append(ao)
    return prunedList


'''
------------------------------------------------------
change ranking baselines
------------------------------------------------------
'''
def generateRandomRanking(changeList): #gets all live/live changed paragraphs TODO: fix duplicates
    
    
    randomList = copy.deepcopy(changeList)
    random.shuffle(randomList)
    return randomList

def generateRankingByTopEdited(livePars, mip, revWindow = 10, sigEdit = True):
    revisionNum = mip.latestVersion
    editCount = {}
    revs = []
    for par in livePars:
        if sigEdit:
            revs = mip.mip.node[par[0]]['sigEditRevs']
        else:
            revs = mip.mip.node[par[0]]['allEditRevs']
        revs.reverse()
        i = 0
        if len(revs)>0:
            while revs[i] > max(0,revisionNum-revWindow):
                i= i+1
                if i==len(revs):
                    break;
        editCount[par[0]] = i
        
    ranking = sorted(editCount.items(), key=operator.itemgetter(1),reverse=True)
    
    return ranking

def generateRankingByTopEditedActionsList(livePars, mip, revWindow = 10, sigEdit = True):
    revisionNum = mip.latestVersion
    editCount = {}
    revs = []
    for par in livePars:
        if sigEdit:
            revs = mip.mip.node[par[0].ao]['sigEditRevs']
        else:
            revs = mip.mip.node[par[0].ao]['allEditRevs']
        revs.reverse()
        i = 0
        if len(revs)>0:
            while revs[i] > max(0,revisionNum-revWindow):
                i= i+1
                if i==len(revs):
                    break;
        editCount[par[0].ao] = i
        
    ranking = sorted(editCount.items(), key=operator.itemgetter(1),reverse=True)
    
    return ranking

def generateRankingByTopEditedActions(actions, mip, revWindow = 10, sigEdit = True):
    revisionNum = mip.latestVersion
    editCount = {}
    revs = []
    for act in actions:
        par = act.ao
        if sigEdit:
            revs = mip.mip.node[par[0]]['sigEditRevs']
        else:
            revs = mip.mip.node[par[0]]['allEditRevs']
        revs.reverse()
        i = 0
        if len(revs)>0:
            while revs[i] > max(0,revisionNum-revWindow):
                i= i+1
                if i==len(revs):
                    break;
        editCount[par[0]] = i
        
    ranking = sorted(editCount.items(), key=operator.itemgetter(1),reverse=True)
    
    return ranking

def generateRankingByRecency(parList, mip, sigEdit = True):
    lastEdit = {}
    revs = []
    for par in parList:
        if sigEdit:
#            print par
            revs = mip.mip.node[par[0]]['sigEditRevs']
        else:
            revs = mip.mip.node[par[0]]['allEditRevs']
        if len(revs)>0:
            lastEdit[par[0]] = revs[len(revs)-1]
        else:
            lastEdit[par[0]] = 0
        
    ranking = sorted(lastEdit.items(), key=operator.itemgetter(1),reverse=True)
    
    return ranking    
            
def generateRankingByRecencyActionList(parList, mip, sigEdit = True):
    lastEdit = {}
    revs = []
    for par in parList:
#        print par
        if sigEdit:
#            print par
            revs = mip.mip.node[par[0].ao]['sigEditRevs']
        else:
            revs = mip.mip.node[par[0].ao]['allEditRevs']
        if len(revs)>0:
            lastEdit[par[0].ao] = revs[len(revs)-1]
        else:
            lastEdit[par[0].ao] = 0
        
    ranking = sorted(lastEdit.items(), key=operator.itemgetter(1),reverse=True)
    
    return ranking           
    

'''
------------------------------------------------------
change ranking baselines end
------------------------------------------------------
'''

'''
------------------------------------------------------
evaluation metrics
------------------------------------------------------
'''
def precisionAtN(actual, predicted, n):
#    print "actual"
#    for change in actual:
#        print change
#    print "predicted"
#    for pred in predicted:
#        print pred
    right = 0.0;

    i = 0;
    while ((i<n) & (i<len(predicted))):
        if predicted[i] in actual:
            right= right+1
        i = i+1

    prec = right/min(n,len(predicted))
    
    prec2 = precision(set(actual), set(predicted[:n]))
   
    
    if (prec!=prec2):
        print 'interesting'         
   
    return prec

def recallAtN(actual, predicted, n):
    retreived = 0.0;
    predPruned = set(predicted[:n])
    for c in actual:
        if c in predPruned:
            retreived= retreived+1
    recall1 = recall(set(actual), set(predicted[:n]))

    return recall1
'''
------------------------------------------------------
evaluation metrics end
------------------------------------------------------
'''
                
def readMIPfromFile(articleName,rev_number):
    pickle_file_name = articleName + "_"+str(rev_number)
    mip_pickle = get_pickle(pickle_file_name, folder = "mip_pickles")
    return mip_pickle
        
    
def generateMIPpicklesForArticles(articleRevisions, articleName, startFrom = 1):
    mip = Mip(articleRevisions.revisions[0])
    mip.initializeMIP()
    pickle_file_name = articleName + "_0"
    mip_file = os.path.join(os.getcwd(), "mip_pickles", pickle_file_name)
    pkl_file = open(mip_file, 'wb')
    print "writing file "+str(pickle_file_name)
    cPickle.dump(mip, pkl_file)
    pkl_file.close()
    for i in range(startFrom,len(articleRevisions.revisions)):
#    for i in range(1,15):

        #try to read from mip, if it already exists no need to recompute
        try: 
            mip_pickle = readMIPfromFile(articleName,i)
#        if (mip_pickle!=None):
#            continue
        except: #mip hasn't yet been computed!
            mip = readMIPfromFile(articleName,i-1)
            mip.updateMIP(articleRevisions.revisions[i])
            pickle_file_name = articleName + "_"+str(i)
            mip_file = os.path.join(os.getcwd(), "mip_pickles", pickle_file_name)
            pkl_file = open(mip_file, 'wb')
            print "writing file "+str(pickle_file_name)
            cPickle.dump(mip, pkl_file)
            pkl_file.close()
        

def runEvalOnArticle(pickle_file_name,generateMIPs = False):
        current_pickle = get_pickle(pickle_file_name)
        print len(current_pickle.revisions)          
        try:
            if generateMIPs == True:        
                generateMIPpicklesForArticles(current_pickle,pickle_file_name[:-4])
                print 'generated all MIPs'
            
        #    results = evaluateChangesForAuthors(current_pickle,"johann_pachelbel")
            results = evaluateGeneralParagraphRankingForAuthors(current_pickle,pickle_file_name[:-4])
            resFileName = "author_predictions_may25/"+ pickle_file_name[:-4] + "_live_par_predictions_sigOnly.csv"
            writeResultsToFile(results, resFileName)
            
            results = evaluateChangesForAuthors(current_pickle,pickle_file_name[:-4])
            resFileName = "author_predictions_may25/"+ pickle_file_name[:-4] + "_changes_predictions_sigOnly.csv"
            writeResultsToFile(results, resFileName)
        except:
            print "Unexpected error:", sys.exc_info()[0]   
            traceback.print_exc() 
        return

def runEvalOnFolder(folderName, generateMIPs = True, removeFiles = True):
        for a in os.walk(folderName):
            for pickle_file_name in a[2]:
                
            
                current_pickle = get_pickle(pickle_file_name)
                print len(current_pickle.revisions)

                try:
                    if generateMIPs == True:        
                        generateMIPpicklesForArticles(current_pickle,pickle_file_name[:-4])
                        print 'generated all MIPs'
                    
                    
                    print 'evaluating all live pars'
                    results2 = evaluateGeneralParagraphRankingForAuthors(current_pickle,pickle_file_name[:-4])
                    resFileName2 = "author_predictions/"+ pickle_file_name[:-4] + "_live_predictions_sigEdits08_simEdge.csv"
                    writeResultsToFile(results2, resFileName2)
                    print 'wrote all live par res to file'
                    
                    print 'evaluating changed pars'
                    results1 = evaluateChangesForAuthors(current_pickle,pickle_file_name[:-4])
                    resFileName1 = "author_predictions/"+ pickle_file_name[:-4] + "_changes_predictions_sigEdits08_simEdge.csv"
                    print 'wrote changed par res to file'
                    
                    writeResultsToFile(results1, resFileName1)
                    
                                    #remove pickle files to save space
                    if (removeFiles):
                        files = glob.glob('mip_pickles/*')
                        for f in files:
                            os.remove(f)
                  
                   
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    traceback.print_exc()
                    

                         
            return 
'''
-----------------------------------------------------------------------------
eval funcs end
----------------------------------------------------------------------------
'''
        


if __name__ == '__main__':

#     runEvalOnFolder('runPickles', False)
    current_pickle = get_pickle("Netherlandish.pkl")
    print len(current_pickle.revisions)
    generateMIPpicklesForArticles(current_pickle,"Netherlandish")

                    
                    
    
#    runEvalOnArticle('Absolute_pitch.pkl', False)
   
    #load necessary data
#    pickle_file_name = 'Absolute_pitch.pkl'
    
#    for a in os.walk('pickles'):
#        for pickle_file_name in a[2]:
#            generate = True
#        
#            current_pickle = get_pickle(pickle_file_name)
#            print len(current_pickle.revisions)
#            
#        #    mip = Mip(current_pickle.revisions[0])
#        #    mip.initializeMIP()
#        #    pickle_file_name = "Yale_University_0"
#        #    mip_file = os.path.join(os.getcwd(), "mip_pickles", pickle_file_name)
#        #    pkl_file = open(mip_file, 'wb')
#        #    print "writing file "+str(pickle_file_name)
#        #    cPickle.dump(mip, pkl_file)
#        #    pkl_file.close()
#            #generate mip pickles
#            try:
#                if generate == True:        
#                    generateMIPpicklesForArticles(current_pickle,pickle_file_name[:-4])
#                    print 'generated all MIPs'
#                
#            #    results = evaluateChangesForAuthors(current_pickle,"johann_pachelbel")
#                results = evaluateGeneralParagraphRankingForAuthors(current_pickle,pickle_file_name[:-4])
#                resFileName = "author_edits_predictions/"+ pickle_file_name[:-4] + "_par_predictions_sigOnly.csv"
#                writeResultsToFile(results, resFileName)
#                
#                results = evaluateChangesForAuthors(current_pickle,pickle_file_name[:-4])
#                resFileName = "author_edits_predictions/"+ pickle_file_name[:-4] + "_changes_predictions_sigOnly.csv"
#                writeResultsToFile(results, resFileName)
#            except:
#                print "Unexpected error:", sys.exc_info()[0]

#    current_texts = current_pickle.get_all_text()
#    current_paras = current_pickle.get_all_paragraphs()
#    print current_paras[0][0].text
#    current_paratexts = [[a.text.encode('utf-8') for a in b] for b in current_paras]
#    revision = current_pickle.revisions
#    print len(current_paratexts[0])
#
#    mip = Mip(revision[0])
#    mip.initializeMIP()
#    print len(revision)
#    for i in range(0,389):
#        mip.updateMIP(revision[i])
#        
#    print '---------mip created----------'  
#    rankedChanges = mip.rankChangesForUser("Nunh-huh", 344, onlySig = False)
#    for change in rankedChanges:
#        print str(change[0])
#        print str(change[1])
#    edgewidth=[]
#    for (u,v,d) in mip.mip.edges(data=True):
#        edgewidth.append(d['weight'])
#
#
#    userNodes = [n for (n,d) in mip.mip.nodes(True) if d['type']=='user']
#    parDeletedNodes = []
#    parNodes = []
#    for (n,d) in mip.mip.nodes(True):
#        if d['type']=='par':
#            if d['deleted']==1:
#                parDeletedNodes.append(n)
#            else:
#                parNodes.append(n)
#
#    pos=nx.spring_layout(mip.mip)
#    new_labels = {}
#    for node,d in mip.mip.nodes(True):
#        if d['type']=='user':
#            new_labels[node]=mip.nodeIdsToUsers[node]
#        else:
#            new_labels[node]=node
#    
#    
#    
#    
##    print DegreeOfInterestMIPs(mip.mip, 3, 7)
#    nx.draw_networkx_nodes(mip.mip,pos,nodelist=userNodes,node_size=300,node_color='red')
#    nx.draw_networkx_nodes(mip.mip,pos,nodelist=parNodes,node_size=300,node_color='blue')
#    nx.draw_networkx_nodes(mip.mip,pos,nodelist=parDeletedNodes, node_size=300,node_color='black')
#    nx.draw_networkx_edges(mip.mip,pos,edgelist=mip.mip.edges(),width=edgewidth)
#    nx.draw_networkx_labels(mip.mip,pos,new_labels)
#    print 'clustering'
#    print(nx.average_clustering(mip.mip, weight = "weight"))
#    #    G=nx.dodecahedral_graph()
##    nx.draw(mip.mip)
#    plt.draw()
##    plt.savefig('ego_graph50.png')
#    plt.show()
    