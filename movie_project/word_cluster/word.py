
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import spacy
import json
import os
import spacy
import json
import numpy as np
from functools import reduce
from sklearn import decomposition
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import nltk.stem as ns

texts = []
for root,dirs,files in os.walk('./script'):
    for file in files:
        with open('./script/{0}'.format(file)) as f:
            texts.append((file[:-4],f.read()))

js = open("./glove.6B.50d_word2id.json", encoding='utf-8')
setting = json.load(js)

for title,text in texts:
    #if n <27:
    #    continue
    print(title)
    sentences = nltk.sent_tokenize(text)
    word_tags = [nltk.pos_tag(nltk.word_tokenize(sent)) for sent in sentences]

    n_list = ['NN','NNP','NNS']
    N =[word[0] for i in range(len(word_tags)) for word in word_tags[i] if word[1] in n_list]

    N = list(set(N))

    v_list = ['VB','VBD','VBG','VBN','VBP','VBZ']
    V= [word[0] for i in range(len(word_tags)) for word in word_tags[i] if word[1] in v_list]



    lemmatizer = ns.WordNetLemmatizer()
    n_lemma = [lemmatizer.lemmatize(word, pos='n') for word in N]
    v_lemma = [lemmatizer.lemmatize(word, pos='v') for word in V]
    n_lemma_lower = [noun.lower() for noun in n_lemma]
    v_lemma_lower = [verb.lower() for verb in v_lemma]

    print(len(v_lemma_lower))


    v_word2id = []
    try:
        for i in v_lemma_lower:
            num = setting.get(i, 0)
            if num != 0:
                v_word2id.append((i,setting[i]))

    except KeyError:
        pass

    print(len(v_word2id))

    #对json里面没有的词进行剔除更新
    try:
        for i in v_lemma_lower:
            num = setting.get(i, 0)
            if num == 0:
                v_lemma_lower.remove(i)

    except KeyError:
        pass

    print(len(v_lemma_lower))



    pre_train = np.load("./glove.6B.50d_mat.npy", allow_pickle=True, encoding="latin1")
    #pre_train = np.loadtxt("./glove.42B.300d.txt", encoding="latin1")
    X = map(lambda x: pre_train[x], [v[1] for v in v_word2id])


    Y = reduce(lambda x, y: np.vstack((x, y)), X)
    print(Y.shape)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Y)
    pos = pd.DataFrame()
    pos['X'] = Z[:, 0]
    pos['Y'] = Z[:, 1]
    #plt.scatter(pos['X'], pos['Y'], )

    estimator = KMeans(n_clusters=5)
    estimator.fit(Z)

    j = 0
    x = []
    y = []
    # 要绘制的点的横纵坐标
    for j in range(len(Z)):
        x.append(Z[j:j + 1, 0])
        y.append(Z[j:j + 1, 1])
    #print(x)
    #print(y)

    label_pred = estimator.labels_  # 获取聚类标签
    centroids = estimator.cluster_centers_  # 获取聚类中心
    inertia = estimator.inertia_  # 获取聚类准则的总和
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # 这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
    plt.figure(figsize=(20, 20))
    color = 0
    j = 0
    for i in label_pred:
        plt.plot(x[j], y[j], mark[i], markersize=5)
        j += 1
    # 为散点打上数据标签
    with open('log.txt', 'a') as f: f.writelines(' '.join(v_lemma_lower))
    v_lemma_lower.reverse()

    print('=' * 20)
    print(len(Z))
    print(len(v_lemma_lower))
    for k in range(len(Z)):
        plt.text(x[k], y[k], v_lemma_lower[k])
    if not os.path.exists('./img'):
        os.mkdir('./img')
    plt.savefig('./img/{0}.jpg'.format(title))
    plt.show()
    plt.close()


js.close()