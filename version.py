import datetime
import Levenshtein


class Paragraph():
    def __init__(self, paratext):
        self.text = paratext
        self.nextindex = None
        self.lastindex = None
        self.changed = False

class Version():
    def __init__(self, text, comment, date, author=''):
        self.text = text
        #self.comment = comment
        self.date = datetime.datetime.strptime(date[:-1], "%Y-%m-%dT%H:%M:%S") #wiki
#        self.date = datetime.datetime.strptime(date[:-1], "%Y%m%dT%H%M%S%f") #gdoc
        self.author = author
        self.paragraphs = []


    def create_paras(self):
        version = self.text.split("\n") #wiki
#        version = self.text.split("\n\n") #gdoc
        indices = range(len(version))
        indices.reverse()
        for paraindex in indices:
            para = version[paraindex]
            #if empty
            if not para:
                version.pop(paraindex)
            elif len(para)<25:
                version.pop(paraindex)
            else:
                #if not on last index
                if not paraindex == len(version)-1:
                    length = len(para)
                    #if last character is not a ., !, or ? or if first char of next paragraph is bullet
                    if (not para[-1] in [('.').encode("UTF-8"),('!').encode("UTF-8"),('?').encode("UTF-8")]) or version[paraindex+1][0] == ('*').encode("UTF-8"):
                        version[paraindex] = version[paraindex] + version[paraindex+1]
                        version.pop(paraindex+1)

        paralist = []
        for a in version:
            paralist.append(Paragraph(a))
        self.paragraphs = paralist




    def showDate(self):
        return self.date

    def __str__(self):
        return 'Author: '+ self.author + '\nDate: ' + str(self.showDate()) + '\nLength: ' + str(len(self.text))\
               + ' paragraphs\nComments: ' + str(len(self.comment))

    def __repr__(self):
        return 'Version('+ str(self.showDate()) +')'

class Page():
    def __init__(self, title):
        self.title = title
        self.revisions = []

    def add_revision(self, text, comment, date, author):
        self.revisions.append(Version(text, comment, date, author))

    def get_all_text(self):
        return [a.text for a in self.revisions]

    def reduce_revisions(self, min_ratio = 15):
        """
        first deletes revisions with not enough text (parsing error)
        deletes all revisions with Levenshtein distance <= a certain value
        """

        self.revisions = [x for x in self.revisions if not len(x.text) < 150]

        ind_to_delete = []
        rev_len = len(self.revisions)
        for ind, rev in enumerate(self.revisions):
            if ind < rev_len-1:
                if Levenshtein.distance(rev.text, self.revisions[ind+1].text) < min_ratio:
                    ind_to_delete.append(ind)

        ind_to_delete.sort(reverse=True)
        for ind in ind_to_delete:
            self.revisions.pop(ind)

    def create_paras(self):
        for rev in self.revisions:
            rev.create_paras()


    def get_all_dates(self):
        return [a.date for a in self.revisions]

    def get_all_authors(self):
        return [a.author for a in self.revisions]

    def get_all_paragraphs(self):
        return [a.paragraphs for a in self.revisions]



