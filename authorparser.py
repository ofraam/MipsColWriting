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
        self.total_authors = None
        self.anon_authors = None
        self.named_authors = None
        self.author_lev_totals = None
        self.author_cumulatives = None
        self.authors_before_eighty = None




#load data
def get_pickle(pickle_file, folder="pickles"):
    pickle_name = os.path.join(os.getcwd(), folder, pickle_file)
    pkl_file = open(pickle_name, 'rb')
    xml_parse = cPickle.load(pkl_file)
    pkl_file.close()
    return xml_parse

   
'''
bunch of similarity functions that return the similarity ratio
'''
    
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



print "done with all the functions"

newer = os.walk("author_predictions").next()[2]
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

            has_authors = False
            try:
                current_names = current_pickle.get_all_authors()
                has_authors = True
            except:
                print "no authors in pickle"
            if has_authors:
                author_levenshtein_totals = {}
                for i1,t1 in enumerate(current_texts):
                    if not i1 == len(current_texts)-1:
                        i2 = i1+1
                        t2 = current_texts[i2]
                        a2 = current_names[i2]
                        dist = Levenshtein.distance(t1, t2)
                        author_levenshtein_totals[a2] = author_levenshtein_totals.get(a2,0) + dist

                author_lev_list = []
                for item, value in author_levenshtein_totals.iteritems():
                    if item:
                        author_lev_list.append(value)
                        
                author_lev_list.sort(reverse=True)

                def get_contribs_before(number_of_a):
                    return np.sum(author_lev_list[:number_of_a+1])

                x = xrange(len(author_lev_list))
                y = [get_contribs_before(a) for a in x]
                s = sum(author_lev_list)
                z = [float(a)/s for a in y]

                eighty = 0
                for a,b in enumerate(z):
                    if b > .8:
                        eighty = b
                        break

                obj.total_authors = len(current_names)
                obj.anon_authors = len([a for a in current_names if a == ''])
                obj.named_authors = len(set(current_names))
                obj.author_lev_totals = author_levenshtein_totals
                obj.author_cumulatives = z
                obj.authors_before_eighty = eighty
                obj.anon_authors_with_sig = authors_sig_changes['']
                obj.named_authors_with_sig = len(authors_sig_changes) -1


                prediction_file = os.path.join(os.getcwd(), "author_predictions", pickle_file_name)
                pkl_file = open(prediction_file, 'wb')
                print "writing file"
                cPickle.dump(obj, pkl_file)
                pkl_file.close()

                print "done"
        



