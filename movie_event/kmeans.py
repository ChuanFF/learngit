import nltk
import os

from nltk.corpus import stopwords

def f1():
    stoplist = stopwords.words('english')
    print(stoplist)
    text = ''

    for root,dirs,files in os.walk('./script/'):
        for file in files:
            with open('./script/{0}'.format(file),'r') as fr:
                for line in fr:
                    text += line.strip(' ')
    '''
    with open('./script/{0}'.format('15-Minutes.txt'),'r') as fr:
        for line in fr:
            text += line.strip(' ')
    '''
    print(text)
    with open('log.txt','w') as f:
        f.write(text)
    sents = nltk.sent_tokenize(text)
    words = []
    for sentce in sents:
        for word in nltk.word_tokenize(sentce):
            if word not in stoplist:
                words.append(word)
    #print(words)
    fdist = nltk.FreqDist(words)
    with open('nltk_log','w') as f:
        for i in fdist.keys():
            f.write(i + ' ')
    print(fdist.keys())

from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
with open('nltk_log','r') as nltk_log:
    from sklearn.externals import joblib
    # word2vec向量化
    model = Word2Vec(LineSentence(nltk_log), size=100, window=5, min_count=1, workers=4)

    # 获取model里面的说有关键词
    keys = model.wv.vocab.keys()

    # 获取词对于的词向量
    wordvector = []
    for key in keys:
        wordvector.append(model[key])

    # 分类
    clf = KMeans(n_clusters=10)
    s = clf.fit(wordvector)
    print(s)
    # 获取到所有词向量所属类别
    labels = clf.labels_

    # 把是一类的放入到一个集合
    classCollects = {}
    for i in range(len(keys)):
        if str(labels[i]) in classCollects.keys():
            classCollects[str(labels[i])].append(list(keys)[i])
        else:
            classCollects[str(labels[i])] = [list(keys)[i]]
    print(type(classCollects))
    print(classCollects)
    with open('sklearn_log.txt','w') as f:
        f.write(str(classCollects))