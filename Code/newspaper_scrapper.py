from lxml import html
import requests
import xattr, biplist
import os
from sklearn.decomposition import LatentDirichletAllocation as LDA
import random
import pandas as pd
import gensim
import nltk
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import ssl
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Read and preprocess data
stemmer = SnowballStemmer('english')
nltk.download('wordnet')

# lemmatizing and stemming
def lem_stem(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text,pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in STOPWORDS and len(token) >= 3:
            result.append(lem_stem(token))
    return result


documents = []

# read saved html files from newspapers
dir_path = '../Data/'
files = os.listdir(dir_path)
count = 0
for file in files:
	if file[-5:] == '.html':
		file_path = dir_path + file
		# x = xattr.xattr(file_path)
		# page = requests.get(biplist.readPlistFromString(x.get('com.apple.metadata:kMDItemWhereFroms'))[0])
		# tree = html.fromstring(x.content)
		tree =  html.parse(file_path)
		texts = tree.xpath('//p[@class="css-158dogj evys1bk0"]/text()') # new york times
		if texts==[]:
			texts = tree.xpath('//p[@class="text-block-container"]/text()') # the star
			if texts==[]:	
				texts = tree.xpath('//div[@class="content__article-body from-content-api js-article__body"]//p/text()') # guardian
				if texts==[]:	
					texts = tree.xpath('//div[@class="story-content"]//p/text()') # national post
					if texts==[]:	
						texts = tree.xpath('//p[@class="bb-p"]/text()') # wired  
						if texts==[]:	
							texts = tree.xpath('//main[@id="content"]//p/text()') # www.theaustralian.com.au 
							if texts==[]:	
								texts = tree.xpath('//div[@class="story-body__inner"]//p/text()') # bbc
								if texts==[]:	
									texts = tree.xpath('//div[@class="article__content"]//p/text()') # the sun
									if texts==[]:	
										texts = tree.xpath('//article//p/text()') # telegraph
										if texts==[]:	
											texts = tree.xpath('//div[@class="story"]//p/text()') # cbc
											if texts==[]:	
												texts = tree.xpath('//div[@class="article-body-container"]//p/text()') # forbes
												if texts==[]:	
													texts = tree.xpath('//div[@class="article-content "]//p/text()') # wall street journal
													if texts==[]:	
														texts = tree.xpath('//div[@class="article-body"]//p/text()') # washington post	
														if texts==[]:	
															texts = tree.xpath('//div[@itemprop="articleBody"]//p/text()') # mailonline
		if texts!=[]:										
			documents.append(' '.join(texts))
			count += 1


documents = pd.Series(documents)
processed_docs = documents.map(preprocess)
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=10, no_above=0.5)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

lda_model = gensim.models.LdaModel(bow_corpus, num_topics=4, alpha='auto',id2word=dictionary)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary, R=20)
pyLDAvis.save_html(vis, 'LDA_Visualization.html')
# vis
