import requests  # 导入requests包
from bs4 import BeautifulSoup
import re
import os


def f1():
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}
    #response = request.get(url,headers=headers)
    url = 'https://www.imsdb.com/genre/Action'
    strhtml = requests.get(url,headers=headers,allow_redirects=False)
    soup = BeautifulSoup(strhtml.text, 'lxml')
    data = soup.select('p')
    print(data)
    #with open("data.txt",'w')as f:
    #    f.write(str(data))
    data_title = re.split("/Movie Scripts/| Script.html",str(data))[1::2]
    for n,i in enumerate(data_title):
        data_title[n] = re.sub("[ ]",'-',data_title[n])
        data_title[n] = re.sub("[:]","",data_title[n])
        data_title[n] = re.sub("&amp;","&",data_title[n])
    with open('title.txt','w') as f:
        for t in data_title:
            f.write(t+'\n')
    print(data_title)

'''
#mainbody > table:nth-child(3) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(3) > p:nth-child(3) > a:nth-child(1)
Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:70.0) Gecko/20100101 Firefox/70.0
#mainbody > table:nth-child(3) > tbody:nth-child(1) > tr:nth-child(1) > td:nth-child(3) > p
html body#mainbody table tbody tr td p
'''

def f2():
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}
    #response = request.get(url,headers=headers)
    pre_url = 'https://www.imsdb.com/scripts/'
    title_list= []
    with open("title.txt",'r') as f:
        for title in f:
            title_list.append(title)

    for title in title_list:
        title = re.sub('[\n]','',title)
        url = pre_url + title + '.html'
        strhtml = requests.get(url,headers=headers,allow_redirects=False)
        soup = BeautifulSoup(strhtml.text, 'lxml')
        print(title)
        print(soup)
        data = soup.select('pre').c
        print('haha')
        print(data)
        try:
            text = list(data)[0].get_text()

        #a2 = data[1]
            if not os.path.exists("script"):
                os.mkdir("script")
            print(data)

            with open('script/{0}.txt'.format(title),'w') as fw:
                fw.write(text)
        except:
            with open('log.txt','a') as fw:
                fw.write(title+'\n')
f2()
