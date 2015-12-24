import cPickle
import os
import re
from version import Page

import bs4 as BS
#regexes precompiled for increased runtime

article_info = re.compile(r":''.*''") #comments by admins

images = re.compile(r"\[\[Image(.*?)(\[\[.*?\]\].*?)*\]\]") #embedded images (can have links in them)
languages = re.compile(r"\[\[(simple|..):.*?\]\]") # languages are either idetified by country code or 'simple'
categories = re.compile(r"\[\[Category:(.*?)\]\]") #category list
lists = re.compile(r"==.*==(\n\*.*)+") #lists of something
long_anchors = re.compile(r"\[\[(.[^\]]*?)\|(.*?)\]\]") #anchors with alternative text
short_anchors = re.compile(r"\[\[(.*?)\]\]") #anchors without alternative text
extra_info = re.compile(r"\{\{(.[^\}]*(\n.[^\}]*)*.*\}\})") #cursive brackets are things like tables and add. infos)
markup_bold_cursive = re.compile(r"''(')?('')?(.*?)('')?(')?''") #bold and cursive text is markes by ''' resp. '' or '''''
html_tags = re.compile(r"<.*?>") #html tags are in the text mostly for cites
headlines = re.compile(r"==(=)?(=)?(.*?)==(=)?(=)?") #headlines are indicated by 2-4 = on each side.
extra_breaks = re.compile(r"\n\n") #due to parsing a lot of empty lines exist


def cleanup_text(curr_text, remove_headlines=True):
    '''
    cleans the text of a wikimedia article
    '''

    curr_text = article_info.sub('', curr_text) #delete comments
    curr_text = images.sub('', curr_text) # delete images
    curr_text = languages.sub('', curr_text) # delete language links
    curr_text = categories.sub('', curr_text) # delete categories of article
    curr_text = lists.sub('', curr_text) # delete 'special' lists (occurences, movies, etc) but not those containing text
    curr_text = long_anchors.sub(lambda x: x.group(2), curr_text) #substitute with alt text
    curr_text = short_anchors.sub(lambda x: x.group(1), curr_text) # substitute with link text
    curr_text = extra_info.sub('', curr_text) # delete tables and charts
    curr_text = markup_bold_cursive.sub(lambda x: x.group(3), curr_text) # delete indicators of bold/cursive text but leave text intact
    curr_text = html_tags.sub('', curr_text) # delete html tags
    if remove_headlines:
        curr_text = headlines.sub('', curr_text) # delete headlines
    for i in range(15):
        curr_text = extra_breaks.sub('\n', curr_text) # delete all blank lines (15 chosen arbitrarily)
    return curr_text

def cleanset(texts):
    return [cleanup_text(texts[i]) for i in range(len(texts))]

def getstuff(xml_name, border1, border2):
    #open the xml file and find all tags with text
    soup = BS.BeautifulSoup(open(os.path.join(os.getcwd(), xml_name)), "lxml")
#    soup = BS.BeautifulSoup(open(xml_name), "lxml")

    #text extraction
    pages = soup.find_all('text')
    texts = [a.text for a in pages]
    texts1 = cleanset(texts[:border1])
    texts1.extend(cleanset(texts[border2:]))

    #additional information
    revision = soup.find_all('revision')
    comments = []
    timestamps = []
    users = []
    for r in revision:
        timestamps.append(r.timestamp.text)
        if r.comment is not None:
            comments.append(r.comment.text)
        else:
            comments.append('')
        if r.contributor.username is not None:
            users.append(r.contributor.username.text)
        else:
            users.append('')

    #strip out the not working parts
    comments1 = comments[:border1]
    comments1.extend(comments[border2:])

    timestamps1 = timestamps[:border1]
    timestamps1.extend(timestamps[border2:])

    users1 = users[:border1]
    users1.extend(users[border2:])

    #get title of page
    title = soup.find('title').text
    #create history object
    page_history = Page(title)
    for i in range(len(texts1)):
        page_history.add_revision(texts1[i], comments1[i], timestamps1[i], users1[i])
    page_history.reduce_revisions()
    page_history.create_paras()
    return page_history

def getstuffOfra(directory_name, border1, border2):
    #open the xml file and find all tags with text
#    soup = BS.BeautifulSoup(open(os.path.join(os.getcwd(), xml_name)), "lxml")
    page_history = Page('colWriting')
    for a in os.walk("papers"):
        for b in a[2]:    
            filename = b 
            values = b.split('_')
            title = values[0]
            revisionNum = values[1]
            timestamp = values[2]
            author = values[3]
            fileToOpen=os.path.join(os.getcwd(), "papers", b)
            with open (fileToOpen, "r") as myfile:
                text=myfile.read()
#                print data 
            page_history.add_revision(text, '',timestamp, author)
#    page_history.reduce_revisions()
    page_history.create_paras()
    return page_history
            
                
            
#
#    #text extraction
#    pages = soup.find_all('text')
#    texts = [a.text for a in pages]
#    texts1 = cleanset(texts[:border1])
#    texts1.extend(cleanset(texts[border2:]))
#
#    #additional information
#    revision = soup.find_all('revision')
#    comments = []
#    timestamps = []
#    users = []
#    for r in revision:
#        timestamps.append(r.timestamp.text)
#        if r.comment is not None:
#            comments.append(r.comment.text)
#        else:
#            comments.append('')
#        if r.contributor.username is not None:
#            users.append(r.contributor.username.text)
#        else:
#            users.append('')
#
#    #strip out the not working parts
#    comments1 = comments[:border1]
#    comments1.extend(comments[border2:])
#
#    timestamps1 = timestamps[:border1]
#    timestamps1.extend(timestamps[border2:])
#
#    users1 = users[:border1]
#    users1.extend(users[border2:])
#
#    #get title of page
#    title = soup.find('title').text
#    #create history object
#    page_history = Page(title)
#    for i in range(len(texts1)):
#        page_history.add_revision(texts1[i], comments1[i], timestamps1[i], users1[i])
#    page_history.reduce_revisions()
#    page_history.create_paras()
#    return page_history


if __name__ == "__main__":
    
    for a in os.walk("xmls"):
        for xml_file in a[2]:
#    xml_file = 'Atom.xml'
    #pickle_file = 'mona_lisa.pkl'
            pickle_file = xml_file[:-4]+ '.pkl'
        
        
            xml_name = os.path.join(os.getcwd(), "xmls", xml_file)
            print xml_name
            pickle_name = os.path.join(os.getcwd(), "pickles", pickle_file)
        
            #in case the user made formatting errors, you need to exclude those revisions
            #do 1000 and 1000 if not
            border1 = 1000
            border2 = 1000
        
        
            #extract from xml
            xml_parse = getstuff(xml_name, border1, border2)
        
            #store as pickle file
            pkl_file = open(pickle_name, 'wb')
            cPickle.dump(xml_parse, pkl_file)
            pkl_file.close()

    #extract from pickle file

    #pkl_file = open(pickle_name, 'rb')
    #xml_parse = cPickle.load(pkl_file)
    #pkl_file.close()
