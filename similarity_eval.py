#imports
import cPickle, copy, custom_texttiling as ct, difflib, gensim, jellyfish, matplotlib.pyplot as plt, matplotlib.cm as cm
import Levenshtein, nltk, nltk.data, numpy as np, os, re
from pylab import gca, Rectangle
from difflib import SequenceMatcher
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from matplotlib import rcParams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#used for splitting function
stop = stopwords.words('english')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

print "starting"

#class for prediction evaluation
class prediction_pickle():
    def __init__(self):
        self.versions= None
        self.same_after = None
        self.all_after = None
        self.all_before_after = None
        self.all_avg_before_after = None
        self.neighbor_after = None
        self.neighbor_before_after_values_indices = None
        self.neighbor_avg_before_after = None
        self.xscatter_initial_n = None
        self.yscatter_neighbor = None
        self.xscatter_initial_a = None
        self.yscatter_all = None
        self.future_linked = None
        self.mappings = None
        self.similarity_linked = None

#load data
def get_pickle(pickle_file, folder="pickles"):
    pickle_name = os.path.join(os.getcwd(), folder, pickle_file)
    pkl_file = open(pickle_name, 'rb')
    xml_parse = cPickle.load(pkl_file)
    pkl_file.close()
    return xml_parse

def sequence_sim(a, b):
    return SequenceMatcher(None, a, b).ratio()

def nltk_sim(a, b):
    return nltk.metrics.edit_distance(a, b)

def jelly_sim(a, b):
    return 1-jellyfish.levenshtein_distance(a, b)/float(max(len(a), len(b)))

def cosine_sim(a, b):
    tfidf_vectorizer = TfidfVectorizer(min_df=1)
    tfidf_matrix_train1 = tfidf_vectorizer.fit_transform((a, b))
    return cosine_similarity(tfidf_matrix_train1[0:1], tfidf_matrix_train1)[0][1]

def lev_sim(a, b):
    return Levenshtein.ratio(a, b)

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



#splits the texts of the paragraphs into sentences. detects punctuation properly using the english pickle
#current_sentences = [[sent_detector.tokenize(a.strip()) for a in para] for para in current_paratexts]

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
    for (key,value) in mf.iteritems():
        if value != None:
            if mb[value] == key:
                new_mappings[key] = value
    return new_mappings



# mapping threshhold function sp1
#take the subset of all mappings where the levenshtein ratio is below 1 (imperfect mappings)
def detect_mapping_changes(t):
    all_mappings_with_changes = []
    all_mappings_without_changes = []
    for i, v in enumerate(all_mappings):
        changed_mappings = {}
        not_changed_mappings = {}
        for key, value in v.iteritems():
            if Levenshtein.ratio(current_paratexts[i][key], current_paratexts[i+1][value]) < t:
                changed_mappings[key] = value
            else:
                not_changed_mappings[key] = value
        all_mappings_with_changes.append(changed_mappings)
        all_mappings_without_changes.append(not_changed_mappings)
    return all_mappings_with_changes, all_mappings_without_changes

#sp2
def get_future_mapping(version,index,future):
    if version + future > versions:
        return None
    while future > 0:
        index = all_mappings[version].get(index)
        if not index:
            return None
        version += 1
        future -=1
    return index

def get_future_mappings(version,index,future):
    try: 
        a = current_paratexts[version][index]
    except: 
        return None
    
    lst = [index]
    while future > 0:
        if version > versions:
            return None
        index = all_mappings[version].get(index)
        if not index:
            return None
        lst.append(index)
        version += 1
        future -=1
    return lst

def get_backward_mapping(version,index,future):
    try: 
        a = current_paratexts[version][index]
    except: 
        return None
    
    lst = [index]
    while future > 0:
        i = None
        for key, value in all_mappings[version-1].iteritems():
            if value == index:
                i = key
        if not i:
            return None
        
        lst.append(i)
        version -= 1
        future -=1
    lst.reverse()
    return lst[0]

def find_patterns_of_no_change(length, specificity):
    dictionary = {}
    mapping_changes = detect_mapping_changes(specificity)[0]
    for v in xrange(len(all_mappings)-1):
        for i in xrange(len(current_paratexts[v])):
            mapped_index = i
            current_version = v
            changed = False
            while not changed:
                if current_version >= len(all_mappings):
                    break
                if mapped_index in mapping_changes[current_version].keys():
                    changed = True
                else:
                    mapped_index = all_mappings[current_version].get(mapped_index)
                    if mapped_index:
                        current_version +=1
                    else:
                        break

            if mapped_index:
                if current_version - v >= length:
                    if not (mapped_index, current_version) in dictionary.keys():
                        dictionary[(mapped_index, current_version)] = (i,v)
            
    return dictionary

#look at future values after each no change point for same paragraph
def get_scores_after_no_change(specificity,length_of_no_change, graph=False, give_indices=False):
    future_values = []
    future_indices = []
    for new,old in find_patterns_of_no_change(length_of_no_change,specificity).iteritems():
        inew=new[0]
        vnew=new[1]
        iold = old[0]
        vold = old[1]
        indices = get_future_mappings(vnew,inew,length_of_no_change)
        if indices:
            array = []
            for offset, paraindex in enumerate(indices):
                if offset < len(indices)-1:
                    array.append(Levenshtein.ratio(current_paratexts[vnew+offset][paraindex], current_paratexts[vnew+offset+1][indices[offset+1]]))
            future_values.append(array)
            if give_indices:
                future_indices.append((inew, vnew))
    
    future_values = np.array(future_values)

    if give_indices:
        return future_values, future_indices
    return future_values

#look at future values after each no change point for all paragraphs in version
def get_all_scores_after_no_change(specificity,length_of_no_change, graph=False, give_indices=False):
    future_values = []
    future_indices = []
    for new,old in find_patterns_of_no_change(length_of_no_change,specificity).iteritems():
        inew=new[0]
        vnew=new[1]
        iold = old[0]
        vold = old[1]
        all_indices = [get_future_mappings(vnew,x,length_of_no_change) for x,y in enumerate(current_paratexts[vnew])]
        
        for k, indices in enumerate(all_indices):
            if indices and not k == inew:
                array = []
                for offset, paraindex in enumerate(indices):
                    if offset < len(indices)-1:
                        array.append(Levenshtein.ratio(current_paratexts[vnew+offset][paraindex], current_paratexts[vnew+offset+1][indices[offset+1]]))
                future_values.append(array)
            if give_indices:
                future_indices.append((inew, vnew))

    future_values = np.array(future_values)

    if give_indices:
        return future_values, future_indices
    return future_values


#look at future values after each no change point for all paragraphs in version
def get_all_scores_before_after_no_change(specificity,length_of_no_change, graph=False, give_indices=False):
    future_values = []
    future_indices = []
    for new,old in find_patterns_of_no_change(length_of_no_change,specificity).iteritems():
        inew=new[0]
        vnew=new[1]
        iold = old[0]
        vold = old[1]
        all_indices = [get_future_mappings(vnew-length_of_no_change,x,length_of_no_change*2) for x,y in enumerate(current_paratexts[vnew])]

        
        for indices in all_indices:
            if indices:
                array = []
                for offset, paraindex in enumerate(indices):
                    if offset < len(indices)-1:
                        array.append(Levenshtein.ratio(current_paratexts[vnew-length_of_no_change+offset][paraindex], current_paratexts[vnew-length_of_no_change+offset+1][indices[offset+1]]))
                future_values.append(array)
            if give_indices:
                future_indices.append((inew, vnew))

    future_values = np.array(future_values)

    if give_indices:
        return future_values, future_indices
    return future_values

#look at future values after each no change point and before for all paragraphs
def get_all_scores_before_after_no_change_avg(specificity,length_of_no_change, graph=False):
    avgs = []
    for new,old in find_patterns_of_no_change(length_of_no_change,specificity).iteritems():
        inew=new[0]
        vnew=new[1]
        iold = old[0]
        vold = old[1]
        all_indices = [get_future_mappings(vnew-length_of_no_change,x,length_of_no_change*2) for x,y in enumerate(current_paratexts[vnew])]
        
        for k, indices in enumerate(all_indices):
            if indices:
                before = 0
                after = 0
                for offset, paraindex in enumerate(indices):
                    if offset < len(indices)-1:
                        if offset < length_of_no_change:
                            before += (Levenshtein.ratio(current_paratexts[vnew-length_of_no_change+offset][paraindex], current_paratexts[vnew-length_of_no_change+offset+1][indices[offset+1]]))
                        else:
                            after += (Levenshtein.ratio(current_paratexts[vnew-length_of_no_change+offset][paraindex], current_paratexts[vnew-length_of_no_change+offset+1][indices[offset+1]]))
                before = before/length_of_no_change
                after = after/length_of_no_change
                avgs.append(before-after)
    if graph:
        plt.hist(avgs, bins=21)
        plt.show()
    
    return avgs

#sp3
#look at future values after each no change point for all paragraphs in version
def get_neighbor_scores_after_no_change(specificity,length_of_no_change, graph=False, give_indices=False):
    future_values = []
    future_indices = []
    for new,old in find_patterns_of_no_change(length_of_no_change,specificity).iteritems():
        inew=new[0]
        vnew=new[1]
        iold = old[0]
        vold = old[1]
        all_indices = [get_future_mappings(vnew,inew+1,length_of_no_change), get_future_mappings(vnew,inew-1,length_of_no_change)]
        
        for indices in all_indices:
            if indices:
                array = []
                for offset, paraindex in enumerate(indices):
                    if offset < len(indices)-1:
                        array.append(Levenshtein.ratio(current_paratexts[vnew+offset][paraindex], current_paratexts[vnew+offset+1][indices[offset+1]]))
                future_values.append(array)
            if give_indices:
                future_indices.append((inew, vnew))
    future_values = np.array(future_values)

    if give_indices:
        return future_values, future_indices
    return future_values


#look at future values after each no change point for all paragraphs in version
def get_neighbor_scores_before_after_no_change(specificity,length_of_no_change, graph=False, give_indices=False):
    future_values = []
    future_indices = []
    for new,old in find_patterns_of_no_change(length_of_no_change,specificity).iteritems():
        inew=new[0]
        vnew=new[1]
        iold = old[0]
        vold = old[1]
        back1 = get_backward_mapping(vnew,inew+1, length_of_no_change)
        back2 = get_backward_mapping(vnew,inew-1, length_of_no_change)
        all_indices = [get_future_mappings(vnew-length_of_no_change,back1,length_of_no_change*2), get_future_mappings(vnew-length_of_no_change,back2,length_of_no_change*2)]
        
        for indices in all_indices:
            if indices:
                array = []
                for offset, paraindex in enumerate(indices):
                    if offset < len(indices)-1:
                        array.append(Levenshtein.ratio(current_paratexts[vnew-length_of_no_change+offset][paraindex], current_paratexts[vnew-length_of_no_change+offset+1][indices[offset+1]]))
                future_values.append(array)
            if give_indices:
                future_indices.append((inew, vnew))

    future_values = np.array(future_values)

    if give_indices:
        return future_values, future_indices
    return future_values


#look at future values after each no change point for all paragraphs in version
def get_neighbor_scores_before_after_no_change_avg(specificity,length_of_no_change, graph=False):
    avgs = []
    for new,old in find_patterns_of_no_change(length_of_no_change,specificity).iteritems():
        inew=new[0]
        vnew=new[1]
        iold = old[0]
        vold = old[1]
        back1 = get_backward_mapping(vnew,inew+1, length_of_no_change)
        back2 = get_backward_mapping(vnew,inew-1, length_of_no_change)
        all_indices = [get_future_mappings(vnew-length_of_no_change,back1,length_of_no_change*2), get_future_mappings(vnew-length_of_no_change,back2,length_of_no_change*2)]            
        for indices in all_indices:
            if indices:
                before = 0
                after = 0
                for offset, paraindex in enumerate(indices):
                    if offset < len(indices)-1:
                        if offset < length_of_no_change:
                            before += (Levenshtein.ratio(current_paratexts[vnew-length_of_no_change+offset][paraindex], current_paratexts[vnew-length_of_no_change+offset+1][indices[offset+1]]))
                        else:
                            after += (Levenshtein.ratio(current_paratexts[vnew-length_of_no_change+offset][paraindex], current_paratexts[vnew-length_of_no_change+offset+1][indices[offset+1]]))
                before = before/length_of_no_change
                after = after/length_of_no_change
                avgs.append(before-after)

    return avgs

#sp4
def plot_scatter_for_neighbors(specificity, length_of_no_change, graph=False):
    initial_change_future_values, initial_change_future_values_indices = get_scores_after_no_change(specificity,length_of_no_change, graph=False, give_indices=True)
    neighbor_change_future_values, neighbor_change_future_values_indices = get_neighbor_scores_after_no_change(specificity,length_of_no_change, graph=False, give_indices=True)
    initial_change_value = [x[0] for x in initial_change_future_values]
    neighbor_change_future_average = [np.mean(x) for x in neighbor_change_future_values]
    x_scatter_initial_change = []
    y_scatter_avg_change_neighbors = []
    for key, value in enumerate(neighbor_change_future_average):
        index_to_search_for = neighbor_change_future_values_indices[key]
        for a, x in enumerate(initial_change_future_values_indices):
            if x == index_to_search_for:
                x_scatter_initial_change.append(initial_change_value[a])
                y_scatter_avg_change_neighbors.append(value)
    if graph:
        plt.scatter(x_scatter_initial_change, y_scatter_avg_change_neighbors)
        plt.title("initial change of paragraph compared to average change of neighbors")
        plt.ylabel("avg change of neighbor")
        plt.xlabel("initial change of paragaraph")
    return (x_scatter_initial_change, y_scatter_avg_change_neighbors)

def plot_scatter_for_all(specificity, length_of_no_change, graph=False):
    initial_change_future_values, initial_change_future_values_indices = get_scores_after_no_change(specificity,length_of_no_change, graph=False, give_indices=True)
    all_change_future_values, all_change_future_values_indices = get_all_scores_after_no_change(specificity,length_of_no_change, graph=False, give_indices=True)
    initial_change_value = [x[0] for x in initial_change_future_values]
    all_change_future_average = [np.mean(x) if np.mean(x) <= 1 else 0 for x in all_change_future_values]
    x_scatter_initial_change = []
    y_scatter_avg_change_all = []
    for key, value in enumerate(all_change_future_average):
        if value:
            index_to_search_for = all_change_future_values_indices[key]
            for a, x in enumerate(initial_change_future_values_indices):
                if x == index_to_search_for:
                    if value != initial_change_value[a]:
                        x_scatter_initial_change.append(initial_change_value[a])
                        y_scatter_avg_change_all.append(value)
    if graph:
        plt.scatter(x_scatter_initial_change, y_scatter_avg_change_all, alpha=.2)
        plt.title("initial change of paragraph compared to average change of all paragraphs")
        plt.ylabel("avg change of all others")
        plt.xlabel("initial change of paragaraph")
        plt.xlim([.55, .9])
    return x_scatter_initial_change, y_scatter_avg_change_all

#sp5
def get_backward_mapping_indefinite(version,index,min_past):
    try: 
        a = current_paratexts[version][index]
    except: 
        return None
    lst = [index]
    broken = False
    past = 0
    while not broken:
        i = None
        if version - 1 >= 0:
            for key, value in all_mappings[version-1].iteritems():
                if value == index:
                    i = key
        if not i:
            broken = True
        else:
            lst.append(i)
            version -= 1
            past +=1
    if past > min_past:
        lst.reverse()
        return lst
    return None

def get_future_mappings_indefinite(version,index,min_future):
    try: 
        a = current_paratexts[version][index]
    except: 
        return None
    
    lst = [index]
    broken = False
    future = 0
    while not broken:
        i = None
        if version < versions:
        
            for key, value in all_mappings[version+1].iteritems():
                if value == index:
                    i = key
        if not i:
            broken = True
        else:
            lst.append(i)
            version += 1
            future +=1
    if future > min_future:
        return lst
    return None
        
def find_significant_changes(threshold):
    significant_changes = detect_mapping_changes(threshold)[0]
    significant_dictionary = {} 
    for key, value in enumerate(significant_changes):
        if value:
            significant_dictionary[key] = value.keys()
    return significant_dictionary

def find_all_backtracks_of_linkings(threshold, min_past):
    sig_changes = find_significant_changes(threshold)
    all_backtracks = {}
    for a in sig_changes.keys():
        sig_change_tracked = False
        all_version_backtracks = {}
        for i in xrange(len(current_paratexts[a])):
            bw = get_backward_mapping_indefinite(a, i, min_past)
            if bw:
                if i in sig_changes[a]:
                    sig_change_tracked = True
                all_version_backtracks[i] = bw
        if all_version_backtracks and len(all_version_backtracks) > 1 and sig_change_tracked:
            all_backtracks[a] = all_version_backtracks
    return all_backtracks, sig_changes
            
def find_backtrack_scores(threshold, min_past):
    backtrack, sig_changes = find_all_backtracks_of_linkings(threshold, min_past)
    all_backtrack_scores = {}
    for version, value in backtrack.iteritems():
        backtrack_version_scores = {}
        for last_index, past_indices in value.iteritems():
            score_array = []
            for i, index in enumerate(past_indices):
                if i < len(past_indices)-1:
                    l_ratio = Levenshtein.ratio(current_paratexts[version-len(past_indices)+i+1][index], current_paratexts[version-len(past_indices)+i+2][past_indices[i+1]])
                else:
                    l_ratio = Levenshtein.ratio(current_paratexts[version-1][index], current_paratexts[version][last_index])
                score_array.append(l_ratio)
            backtrack_version_scores[last_index] = score_array
        all_backtrack_scores[version] = backtrack_version_scores
    return backtrack, sig_changes, all_backtrack_scores

#if sig change happens, detect which paragraphs are linked in that version
def find_para_linked_to_sig_change(sig_change_threshold, min_past,change_occured_threshold, link_threshold):    
    backtrack, sig_changes, all_backtrack_scores = find_backtrack_scores(sig_change_threshold, min_past)
    linked_paragraphs_by_version = {}
    for version, indices in sig_changes.iteritems():
        if version in all_backtrack_scores.keys():
            list_of_links = []
            for index in indices:
                if not index in all_backtrack_scores[version].keys():
                    break
                iscores = all_backtrack_scores[version][index]
                iscores = [0 if x < change_occured_threshold else 1 for x in iscores]
                for compare,cscores in all_backtrack_scores[version].iteritems():
                    if not compare == index:
                        iters = min(len(iscores),len(cscores))
                        cscores = [0 if x < change_occured_threshold else 1 for x in cscores]
                        new_iscores = iscores[-iters:]
                        cscores = cscores[-iters:]
                        if not len(new_iscores) == len(cscores):
                            print "not the same length, oh no"
                        iscores_changes = len(new_iscores) - sum(new_iscores)
                        count = 0
                        for j, item in enumerate(cscores):
                            if item == 0 and new_iscores[j] == 0:
                                count +=1
                        if iscores_changes == 0:
                            avg = 0
                        else:
                            avg = float(count)/iscores_changes
                        if avg > link_threshold:
                            '''if compare < index:
                                if not (compare,index) in list_of_links:
                                    list_of_links.append((compare,index))
                            else:
                                if not (index,compare) in list_of_links:
                                    list_of_links.append((index,compare))'''
                            list_of_links.append((index,compare))

            if list_of_links:
                linked_paragraphs_by_version[version] = list_of_links
    return linked_paragraphs_by_version

# determine if linked paras change together in the future
def find_future_of_linked_paras(sig_change_threshold, min_past,change_occured_threshold, link_threshold, n_into_future_min, n_into_future_max):    
    #find sig changes linked paras to validate in future
    linked_paras_with_sig_change = find_para_linked_to_sig_change(sig_change_threshold,min_past,change_occured_threshold,link_threshold)
    total_prev_links = 0
    found_link = 0
    looked_at = 0
    #construct new dictionary to validate
    validation_dictionary = copy.deepcopy(linked_paras_with_sig_change)
    for v in linked_paras_with_sig_change:
        for i,link in enumerate(linked_paras_with_sig_change[v]):
            total_prev_links += 1
            t = None
            get_fm = get_future_mappings_indefinite(v,link[0],n_into_future_min)
            get_fm2 = get_future_mappings_indefinite(v,link[1],n_into_future_min)
            if not (get_fm and get_fm2):
                validation_dictionary[v][i] = (linked_paras_with_sig_change[v][i],False)
            else:
                looked_at+=1
                min_len = min(len(get_fm), len(get_fm2), n_into_future_max+1)
                get_fm = get_fm[:min_len]
                get_fm2 = get_fm2[:min_len]
                t = (get_fm[-1],get_fm2[-1])
                
                linked = False
                #calculate if still linked (2 dimensional??)
                change_scores1 = []
                change_scores2 = []
                #version is v
                #get_fmX is indices      
                for t, para_index in enumerate(get_fm):
                    if t< len(get_fm)-1: 
                        change_scores1.append(Levenshtein.ratio(current_paratexts[v+t][para_index], current_paratexts[v+t+1][get_fm[t+1]]))
                for t, para_index in enumerate(get_fm2):
                    if t< len(get_fm2)-1: 
                        change_scores2.append(Levenshtein.ratio(current_paratexts[v+t][para_index], current_paratexts[v+t+1][get_fm2[t+1]]))
                
                num_first_changes = 0
                for change in change_scores1:
                    if change < change_occured_threshold:
                        num_first_changes += 1
                num_second_changes = 0
                for t, change in enumerate(change_scores2):
                    if change < change_occured_threshold and change_scores1[t] < change_occured_threshold:
                            num_second_changes += 1
                avg = 0
                if num_first_changes == 0 and num_second_changes == 0:
                    avg = 1.0
                elif num_first_changes > 0:
                    avg = float(num_second_changes)/num_first_changes
                        
                #only do from current verision to this n versions away
                if avg > link_threshold:
                    validation_dictionary[v][i] = (linked_paras_with_sig_change[v][i],avg)
                    found_link +=1
                else: 
                    validation_dictionary[v][i] = (linked_paras_with_sig_change[v][i],False, avg)
    return validation_dictionary, (found_link, looked_at, total_prev_links)

def find_sig_change_links(sig_change_threshold, linked_change_threshold):
    curr_dict = {}
    filtered_dict = {}
    sigs = find_significant_changes(sig_change_threshold)
    for version, paras in sigs.iteritems():
        for value in paras:
            beginning = version_para_to_index[(version,0)]
            end = version_para_to_index[(version,len(current_paratexts[version])-1)]
            curr_dict[(version,value)] = []
            list_of_filtered = []
            for current in xrange(beginning, end+1):
                vec_bow=dictionary.doc2bow(current_paratexts[version][value].lower().split())
                vec_lsi= lsi[vec_bow]
                score = index[vec_lsi]
                curr_dict[(version,value)].append((current,score[current]))
                if score[current] > linked_change_threshold and not value == current-beginning:
                    list_of_filtered.append((current-beginning,score[current]))
            if list_of_filtered:
                filtered_dict[(version, value)] = list_of_filtered
    return (curr_dict, filtered_dict)

# determine if linked paras change together in the future
def find_future_of_linked_paras_similarity(sig_change_threshold,linked_change_threshold, change_occured_threshold, link_threshold, n_into_future_min, n_into_future_max):    
    filtered_change_links_by_similarity = find_sig_change_links(sig_change_threshold, linked_change_threshold)[1]

    total_prev_links = 0
    found_link = 0
    looked_at = 0
    #construct new dictionary to validate
    validation_list = []
    for k,values in filtered_change_links_by_similarity.iteritems():
        for i,link in enumerate(values):
            total_prev_links += 1
            t = None
            get_fm = get_future_mappings_indefinite(k[0],k[1],n_into_future_min)
            get_fm2 = get_future_mappings_indefinite(k[0],link[0],n_into_future_min)
            if not (get_fm and get_fm2):
                validation_list.append((k[0],k[1],link[0],False))
            else:
                looked_at+=1
                min_len = min(len(get_fm), len(get_fm2), n_into_future_max+1)
                get_fm = get_fm[:min_len]
                get_fm2 = get_fm2[:min_len]
                t = (get_fm[-1],get_fm2[-1])
                
                linked = False
                #calculate if still linked (2 dimensional??)
                change_scores1 = []
                change_scores2 = []
                #version is v
                #get_fmX is indices
                
                
                for t, para_index in enumerate(get_fm):
                    if t< len(get_fm)-1: 
                        change_scores1.append(Levenshtein.ratio(current_paratexts[k[0]+t][para_index], current_paratexts[k[0]+t+1][get_fm[t+1]]))
                for t, para_index in enumerate(get_fm2):
                    if t< len(get_fm2)-1: 
                        change_scores2.append(Levenshtein.ratio(current_paratexts[k[0]+t][para_index], current_paratexts[k[0]+t+1][get_fm2[t+1]]))
                
                num_first_changes = 0
                for change in change_scores1:
                    if change < change_occured_threshold:
                        num_first_changes += 1
                num_second_changes = 0
                for t, change in enumerate(change_scores2):
                    if change < change_occured_threshold and change_scores1[t] < change_occured_threshold:
                            num_second_changes += 1
                avg = 0
                if num_first_changes == 0 and num_second_changes == 0:
                    avg = 1.0
                elif num_first_changes > 0:
                    avg = float(num_second_changes)/num_first_changes
                
                        
                #only do from current verision to this n versions away
                
                if avg > link_threshold:
                    validation_list.append((k[0], k[1], link[0],avg))
                    found_link +=1
                else: 
                    validation_list.append((k[0], k[1], link[0],False, avg))
    
    
    return validation_list, (found_link, looked_at, total_prev_links)



print "done with all the functions"
newer = os.walk("linked").next()[2]
old_pickles = os.walk("newer_predictions").next()[2]

#cell for storing pickles sp6
for a in os.walk("pickles"):
    for b in a[2]:
        if b not in newer and not "ofra" in b:
            print b

            #sp0
            #load necessary data
            pickle_file_name = b
            current_pickle = get_pickle(pickle_file_name)
            current_texts = current_pickle.get_all_text()
            current_paras = current_pickle.get_all_paragraphs()
            current_paratexts = [[a.text.encode('utf-8') for a in b] for b in current_paras]
            obj = prediction_pickle()
            sig_change_threshold = .8
            length_of_no_change_threshold = 10
            min_past = 5
            change_occured_threshold = .8
            link_threshold = .4
            n_into_future_min = 4
            n_into_future_max = 10
            linked_change_threshold = .2

            print "done with setting thresholds"

            try:
                helperpickle = get_pickle(pickle_file_name, "newer_predictions")
                all_mappings = helperpickle.mappings
            except:


                #generate a list of all mappings, threshold was empirically chosen as .4
                all_mappings = []
                for i in xrange(len(current_paratexts)):
                    if i < len(current_paratexts)-1:
                        m_f = assign_neighbors(current_paratexts[i], current_paratexts[i+1], .4)
                        m_b = assign_neighbors(current_paratexts[i+1], current_paratexts[i], .4)
                        all_mappings.append(generate_mapping(m_f, m_b))



            print "did all_mappings"

            versions = len(all_mappings) -1

            #1 - tokenize the text. version_para_to_index will get used to keep track of indices because we add all paragraphs to a long list
            list_of_paratexts = []
            iterator = 0
            version_para_to_index = {}
            for i in xrange(len(current_paratexts)):
                pararange = xrange(len(current_paratexts[i]))
                for j in pararange:
                    list_of_paratexts.append(current_paratexts[i][j])
                    version_para_to_index[(i,j)] = iterator
                    iterator += 1
            print "found version para to index"

            #1.1 add all words to a list and throw stopwords away. stop is a list of stopwords (defined at top)
            list_of_filtered_paratexts = [[word for word in document.lower().split() if word not in stop]
                     for document in  list_of_paratexts]
            #1.2 filter out http from paragraphs
            list_of_filtered_paratexts = [[word for word in doc if 'http:' not in word] for doc in list_of_filtered_paratexts]
            print "found filtered paratexts"

            #1.3 - tokenize the text. use only words that occur more than once
            all_tokens = sum(list_of_filtered_paratexts, [])
            tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
            texts = [[word for word in text if word not in tokens_once]
                     for text in list_of_filtered_paratexts]
            print "finished tokens once and text"

            #2. create a dictionary from the tokens (assign an ID to each word)
            dictionary = corpora.Dictionary(texts)
            print "built corpus"

            #3. transform dict to vector and then to Tf-idf
            corpus = [dictionary.doc2bow(text) for text in texts]
            tfidf = models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]
            print "built tfidf"

            #4. transform the Tf-idf into a LSI-topic model
            lsi = models.LsiModel(corpus_tfidf, id2word=dictionary) # initialize an LSI transformation
            corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
            index = similarities.MatrixSimilarity(lsi[corpus])
            print "did lsi indexing"

            '''obj.mappings = all_mappings

            obj.versions = versions
            print "1"
            obj.same_after = helperpickle.same_after #get_scores_after_no_change(sig_change_threshold, length_of_no_change_threshold)

            print "2"
            obj.all_after = helperpickle.all_after #get_all_scores_after_no_change(sig_change_threshold, length_of_no_change_threshold)
            print "3"
            obj.all_before_after = helperpickle.all_before_after #get_all_scores_before_after_no_change(specificity = sig_change_threshold,length_of_no_change = length_of_no_change_threshold, graph=False, give_indices=False)
            print "4"
            obj.all_avg_before_after = helperpickle.all_avg_before_after #get_all_scores_before_after_no_change_avg(specificity= sig_change_threshold,length_of_no_change = length_of_no_change_threshold, graph=False)
            print "5"
            obj.neighbor_after = helperpickle.neighbor_after #get_neighbor_scores_after_no_change(specificity = sig_change_threshold,length_of_no_change = length_of_no_change_threshold, graph=False, give_indices=True)
            print "6"
            obj.neighbor_before_after_values_indices = helperpickle.neighbor_before_after_values_indices #get_neighbor_scores_before_after_no_change(specificity = sig_change_threshold,length_of_no_change = length_of_no_change_threshold, graph=False, give_indices=True)
            print "7"
            obj.neighbor_avg_before_after = helperpickle.neighbor_avg_before_after #get_neighbor_scores_before_after_no_change_avg(specificity = sig_change_threshold,length_of_no_change = length_of_no_change_threshold, graph=False)
            print "8"
            obj.xscatter_initial_n, obj.yscatter_neighbor = helperpickle.xscatter_initial_n, helperpickle.yscatter_neighbor #plot_scatter_for_neighbors(specificity = sig_change_threshold, length_of_no_change = length_of_no_change_threshold, graph=False)
            print "9"
            obj.xscatter_initial_a, obj.yscatter_all = helperpickle.xscatter_initial_a, helperpickle.yscatter_all #plot_scatter_for_all(specificity = sig_change_threshold, length_of_no_change = length_of_no_change_threshold, graph=False)
            print "10" '''
            obj.similarity_linked = find_future_of_linked_paras_similarity(sig_change_threshold,linked_change_threshold, change_occured_threshold, link_threshold, n_into_future_min, n_into_future_max)


            prediction_file = os.path.join(os.getcwd(), "linked", pickle_file_name)
            pkl_file = open(prediction_file, 'wb')
            print "writing file"
            cPickle.dump(obj, pkl_file)
            pkl_file.close()

            print "done"


