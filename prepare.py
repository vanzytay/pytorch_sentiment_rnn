#!/usr/bin/env python
# Script to prepare datasets
from xml.dom import minidom
import xml.etree.cElementTree as ET
from utilities import *
from keras.preprocessing import sequence
import numpy as np
from tqdm import tqdm
import string
import nltk
import os
import re

dataset = 'Restaurants'
mode = 'term'
train_sentences = []
test_sentences = []
aspectTerms = []
aspectCats = []
words = []
toy = False
one_hot = False

from string import punctuation
def strip_punctuation(s):
	return s.translate(string.maketrans("",""), string.punctuation)
    # return ''.join(c for c in s if c not in punctuation)

def process_text(x):
	x = x.lower()
	x = re.sub('[^A-Za-z0-9]+', ' ', x)
	x = x.split(' ')
	x = [strip_punctuation(y) for y in x]
	# ptxt = nltk.word_tokenize(ptxt)
	return x


# Parse Training File 
tree = ET.ElementTree(file='./datasets/{}_Train.xml'.format(dataset))
for index, sentence in enumerate(tree.iter(tag='sentence')):
	s = {}
	for elem in sentence.iter():
		if(elem.tag=='text'):
			# ptxt = strip_punctuation(elem.text.lower())
			# # ptxt = nltk.word_tokenize(ptxt)
			# ptxt = ptxt.split(' ')
			ptxt = process_text(elem.text)

			s['text'] = ptxt
			words += ptxt
		elif(elem.tag=='aspectTerms' and mode=='term'):
			s['aspectTerms'] = []
			for at in elem.iter():
				attr = at.attrib
				if('term' not in attr):
					continue
				txt = process_text(at.attrib['term'])
				print(txt)
				words += txt
				s['aspectTerms'].append([txt, at.attrib])
				aspectTerms.append(txt)
		elif(elem.tag=='aspectCategories' and mode=='aspect'):
			s['aspectCats'] = []
			for ac in elem.iter():
				attr = ac.attrib
				if('category' not in attr):
					continue
				s['aspectCats'].append([attr['category'],attr])
			aspectCats.append(attr['category'])
	train_sentences.append(s)

all_text = []

# Parse Testing File
tree = ET.ElementTree(file='./datasets/{}_Test.xml'.format(dataset))
for index, sentence in enumerate(tree.iter(tag='sentence')):
	s = {}
	for elem in sentence.iter():
		if(elem.tag=='text'):
			ptxt = process_text(elem.text)
			all_text.append(ptxt)
			s['text'] = ptxt
			words += ptxt
		elif(elem.tag=='aspectTerms' and mode=='term'):
			s['aspectTerms'] = []
			for at in elem.iter():
				attr = at.attrib
				if('term' not in attr):
					continue
				txt = process_text(at.attrib['term'])
				print(txt)
				words += txt
				s['aspectTerms'].append([txt, at.attrib])
				aspectTerms.append(txt)
		elif(elem.tag=='aspectCategories' and mode=='aspect'):
			s['aspectCats'] = []
			for ac in elem.iter():
				attr = ac.attrib
				if('category' not in attr):
					continue
				s['aspectCats'].append([attr['category'],attr])
	test_sentences.append(s)

# aspectTerms = list(set(aspectTerms))
aspectCats = list(set(aspectCats))
if(mode=='term'):
	term_lens = [len(x) for x in aspectTerms]

# words += aspectTerms 
words += aspectCats
words = list(set(words))
all_lens = [len(x) for x in all_text]
max_len, avg_len, min_len = np.max(all_lens), np.mean(all_lens),np.min(all_lens)
if(mode=='term'):
	max_term_len = np.max(term_lens)
	print("{} aspect terms".format(len(aspectTerms)))
	print("{} max len for aspect terms".format(np.max(term_lens)))
if(mode=='aspect'):
	print("{} aspect categories".format(len(aspectCats)))
print("{} unique words".format(len(words)))
print("{} train sentences".format(len(train_sentences)))
print("{} test sentences".format(len(test_sentences)))
print("max sent={} avg sent={} min sent={}".format(max_len, avg_len, min_len))

# Building vocab indices
index_word = {index+2:word for index,word in enumerate(words)}
word_index = {word:index+2 for index,word in enumerate(words)}
index_word[0], index_word[1] = '<pad>','<unk>'
word_index['<pad>'], word_index['unk'] = 0,1
if(mode=='aspect'):
	index_cat = {index:word for index, word in enumerate(aspectCats)}
	cat_index = {word:index for index, word in enumerate(aspectCats)}

# Avoid using 0 incase we want to pad zero vectors
sentiment_map = {
	'positive':2,
	'neutral':1,
	'negative':0
}

def split_terms(tokens, terms):
	''' Split tokens based on terms
	Returns useful meta information regarding positions etc.
	'''
	if(len(terms)==1):
		# split at term
		term = terms[0]
		if(term not in tokens):
			return None
		start_pos = tokens.index(term)
		end_pos = start_pos
		left = tokens[:end_pos+1]
		right = tokens[start_pos:]
	else:
		start = terms[0]
		end = terms[-1]
		if(start not in tokens or end not in tokens):
			print('====================')
			print(tokens)
			print([index_word[x] for x in tokens])
			print([index_word[x] for x in terms])
			return None
		start_pos = tokens.index(start)
		end_pos = tokens.index(end)
		# Overlap the terms
		left = tokens[:end_pos+1]
		right = tokens[start_pos:]
	return [[left, right], [start_pos, end_pos]]
		
def make_terms(txt, pair):
	attr, value = pair[0],pair[1]
	term_len = len(attr)
	tokenized_terms = [word_index[x] for x in attr]

	target_positions = split_terms(txt, tokenized_terms)
	if(target_positions is None):
		print("[Warning] Target not found in text!")
		return None
	# tokenized_terms = sequence.pad_sequences([tokenized_terms],maxlen=max_term_len)[0]
	polarity = value['polarity']
	if(polarity=='conflict'):
		return None
	polarity = sentiment_map[polarity]
	if(one_hot==True):
		vec = np.zeros(3)
		vec[polarity] = 1
		polarity = vec
	tmp = {
	'tokenized_txt':tokenized_txt,
	'actual_len':actual_len,
	'term_id':tokenized_terms,
	'term_len':term_len,
	'polarity':polarity,
	'left':target_positions[0][0],
	'right':target_positions[0][1],
	'pointers':target_positions[1]
	}
	# tmp = [tokenized_txt, actual_len, tokenized_terms, term_len, polarity]
	return tmp

def make_aspects(txt, pair):
	attr, value = pair[0],pair[1]
	attr_id = cat_index[attr]
	polarity = value['polarity']
	if(polarity=='conflict'):
		return None
	polarity = sentiment_map[polarity]
	if(one_hot==True):
		vec = np.zeros(3)
		vec[polarity] = 1
		polarity = vec
	tmp = {
	'tokenized_txt':tokenized_txt,
	'actual_len':actual_len,
	'aspect_id':[attr_id],
	'aspect_len':1,
	'polarity':polarity
	}
	# tmp = [tokenized_txt, actual_len, [attr_id], 0, polarity]
	return tmp

# Converting to Model Friendly Format (Categories)
training_set, testing_set = [], []
if(toy==True):
	print("Using TOY mode")
	train_sentences = train_sentences[:500]
	test_sentences = test_sentences[:100]
for sent in train_sentences:
	txt = sent['text']
	tokenized_txt = [word_index[x] for x in txt]
	actual_len = len(tokenized_txt)
	# tokenized_txt = sequence.pad_sequences([tokenized_txt],maxlen=max_len)[0]
	if('aspectCats' in sent and mode=='aspect'):
		for pair in sent['aspectCats']:
			tmp = make_aspects(tokenized_txt, pair)
			if(tmp is not None):
				training_set.append(tmp)
			training_set.append(tmp)
	elif('aspectTerms' in sent and mode=='term'):
		for pair in sent['aspectTerms']:
			tmp = make_terms(tokenized_txt, pair)
			if(tmp is not None):
				training_set.append(tmp)

for sent in test_sentences:
	txt = sent['text']
	tokenized_txt = [word_index[x] for x in txt]
	actual_len = len(tokenized_txt)
	# tokenized_txt = sequence.pad_sequences([tokenized_txt],maxlen=max_len)[0]
	if('aspectCats' in sent):
		for pair in sent['aspectCats']:
			tmp = make_aspects(txt, pair)
			if(tmp is not None):
				testing_set.append(tmp)
			testing_set.append(tmp)
	elif('aspectTerms' in sent):
		for pair in sent['aspectTerms']:
			tmp = make_terms(tokenized_txt, pair)
			# print("Adding to test set")
			if(tmp is not None):
				testing_set.append(tmp)

print("Partioning into Dev Set")
import random
# random.shuffle(training_set)
# dev_set = training_set[:500]
dev_set = []
# training_set = training_set[500:]

env = {
	'index_word':index_word,
	'word_index':word_index,
	'train':training_set,
	'dev':dev_set,
	'test':testing_set,
	'max_len':max_len
}

if(mode=='aspect'):
	env['index_cat'] = index_cat
	env['cat_index'] = cat_index
elif(mode=='term'):
	env['term_len'] = max_term_len

print("Training set={}".format(len(training_set))) # actual->3518
print("Development Set={}".format(len(dev_set)))
print("Testing set={}".format(len(testing_set))) # actual->973

glove = {}
import cPickle as pickle

dimensions = 300

glove_path = './embeddings/glove_{}_{}.pkl'.format(dataset,mode)
if(os.path.isfile(glove_path)):
	print("Reusing glove dictionary to save time")
	with open(glove_path,'r') as f:
		glove = pickle.load(f)
	save = False
else:
	# Load word embeddings
	with open('../glove_embeddings/glove.840B.{}d.txt'.format(dimensions),'r') as f:
		lines = f.readlines()
		for l in tqdm(lines):
			vec = l.split(' ')
			word = vec[0].lower()
			vec = vec[1:]
			#print(word)
			#print(len(vec))
			glove[word] = np.array(vec)

	print('glove size={}'.format(len(glove)))
	save = True
	
	print("Finished making glove dictionary")

matrix = np.zeros((2, dimensions))
#print(matrix.shape)

oov = 0 

filtered_glove = {}
for i in tqdm(range(2, len(word_index))):
	word = index_word[i]
	if(word in glove):
		vec = glove[word]
		if(save==True):
			filtered_glove[word] = glove[word]
		# print(vec.shape)
		matrix = np.vstack((matrix,vec))
	else:
		random_init = np.random.uniform(low=-0.01,high=0.01, size=(1,dimensions))
		# print(random_init)
		matrix = np.vstack((matrix,random_init))
		oov +=1
		# print(word)

if(save==True):
	with open(glove_path,'w+') as f:
		pickle.dump(filtered_glove, f)
	print("Saving glove dict to file")

print(matrix.shape)
print(len(word_index))
print("oov={}".format(oov))

print("Saving glove vectors")
env['glove'] = matrix



if(toy):
	file_path = './store/{}_{}_toy.pkl'.format(dataset, mode)
else:
	file_path = './store/{}_{}.pkl'.format(dataset, mode)
with open(file_path,'w+') as f:
	pickle.dump(env, f)



# dictToFile(env,'./store/{}_Cat.json.gz'.format(dataset))









