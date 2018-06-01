import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup


###### inputs raw text, outputs split string lemmatized, punctuation removed, stop words removed.
def clean_str(string):
    """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
    stopWords = set(stopwords.words('english'))
    lemma= WordNetLemmatizer()
    #string=string.lower()
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #string = re.sub(r"\'s", " \'s", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"n\'t", " n\'t", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    #newstring=''
    #for char in string:
    #    if (char ==' ') or (char>='0' and char<='9')  or (char>='A' and char<='Z') or (char>='a' and char<='z') or (char=='+') or (char=='#'):
    #        newstring=newstring+char
    #newstring=newstring.split()
    #sentarray=[w for w in newstring if not w in stopWords]
    #for count in xrange(len(sentarray)):
    #    sentarray[count]=str(lemma.lemmatize(sentarray[count]))
    sentarray=re.sub(r"[^a-z0-9#+ ]", "", string.lower()).split()
    sentarray=[w for w in sentarray if not w in stopWords]
    sentarray=[lemma.lemmatize(w) for w in sentarray]
    return sentarray

def clean_str_no_l(string):
    """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
    stopWords = set(stopwords.words('english'))
    #string=string.lower()
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #string = re.sub(r"\'s", " \'s", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"n\'t", " n\'t", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    #newstring=''
    #for char in string:
    #    if (char ==' ') or (char>='0' and char<='9')  or (char>='A' and char<='Z') or (char>='a' and char<='z') or (char=='+') or (char=='#'):
    #        newstring=newstring+char
    #newstring=newstring.split()
    #sentarray=[w for w in newstring if not w in stopWords]
    #for count in xrange(len(sentarray)):
    #    sentarray[count]=str(lemma.lemmatize(sentarray[count]))
    sentarray=re.sub(r"[^a-z0-9#+ ]", " ", string.lower()).split()
    sentarray=[w for w in sentarray if not w in stopWords]
    return sentarray
def extracttext_nc2(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('table'):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    codes=[]
    for c in soup.find_all('code'):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc9(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('pre'):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    codes=[]
    for c in soup.find_all('span'):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc7(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('pre', attrs={'class':'prettyprint lang-config'}):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc_d8(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('div', attrs={'class':'w3-code'}):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc3(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('code'):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc10(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('table'):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc8(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('pre'):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc4(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('span'):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc6(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('div', attrs={'class':'source'}):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc5(arg):
    soup = arg
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('b'):
        text=c.get_text(' ')
        if len(text.split())>2:
            codes.append(text)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc(arg):
    soup = BeautifulSoup(arg, 'html.parser')
    post=unicode(soup.get_text(' '))
    codes=[]
    for c in soup.find_all('code'):
        codes.append(c.string)
    for i in codes:
        if i!=None:
            post=post.replace(i,u' ',1)
    return post
def extracttext_nc_gen(html,filts):
    soup = html
    post=unicode(soup.get_text(' '))
    for a in filts:
        if a!=None:
            codes=[]
            for c in soup.find_all(a):
                text=c.get_text(' ')
                if len(text.split())>2:
                    codes.append(text)
            for i in codes:
                if i!=None:
                    post=post.replace(i,u' ',1)
    return post
def extracttext_nc_gen1(html,filts):
    soup = html
    post=unicode(soup.get_text(' '))
    for a in filts:
        if a!=None:
            codes=[]
            for c in soup.find_all(a):
                text=c.get_text()
                if len(text.split())>2:
                    codes.append(text)
            for i in codes:
                if i!=None:
                    post=post.replace(i,u' ',1)
    return post
def extracttext(arg):
    end=arg.find('</code>')
    start=arg.rfind('<code>',0, end)
    while start !=-1 and end !=-1:
        arg=arg[:start]+arg[end+7:]
        end=arg.find('</code>')
        start=arg.rfind('<code>',0, end)
    start=arg.find('<')
    end=arg.find('>')
    startig=0
    endig=0
    while start !=-1 and end !=-1:
        if start<end:
            if arg[start+1:end].find('<')==-1:
                arg=arg[:start]+' '+arg[end+1:]
            else:
                startig=(start)
        else:
            endig=end
        start=arg.find('<',startig+1)
        end=arg.find('>',endig+1)
    return arg

def maketermlist(tagsdict):
    print tagsdict
    td={}
    max=0
    maxterm=''
    termlist=[]
    for i in xrange(len(tagsdict)):
        if tagsdict.values()[i]>9:
            td[tagsdict.keys()[i]]=tagsdict.values()[i]
    qqqq= len(td)
    print len(tagsdict)
    print qqqq
    for z in xrange(qqqq/3):
        for i in xrange(len(td)):
            if i%5000==0:
                print 'i= ' + str(i)
            if td.values()[i]>max:
                max=td.values()[i]
                maxterm=td.keys()[i]
        termlist.append(maxterm)
        del td[maxterm]
        max=0
        maxterm=''
    print termlist
    return termlist







