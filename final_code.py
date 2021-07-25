
# coding: utf-8

# ### 1. Importing important libraries
import numpy as np
import PyPDF2
import sys

import matplotlib.pyplot as plt

import networkx as nx

# the PunktSentenceTokenizer library is being imported from the file punkt.py contained in package nltk.tokenize 
# this is used to tokenize the document into sentences

from nltk.tokenize.punkt import PunktSentenceTokenizer

# tFidTransformer: In this implementation,TfidfTransformer is used for executing the method fit_transform()... 
# which provides the output as a document-term matrix normalized (value 0-1) according to the TF-IDF

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# ### 2.  Function to read the document from user
def readDoc():
    name = input('Please input a file name: ') 
    print('You have asked for the document {}'.format(name))

    if name.lower().endswith('.txt'):
        choice = 1
    elif name.lower().endswith('.pdf'):
        choice = 2
    else:
        choice = 3

    print(choice)
    # Case 1: if it is a .txt file
        
    if choice == 1:
        f = open(name, 'r')
        document = f.read()
        f.close()
            
    # Case 2: if it is a .pdf file
    elif choice == 2:
        pdfFileObj = open(name, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pageObj = pdfReader.getPage(0)
        document = pageObj.extractText()
        pdfFileObj.close()
    
    # Case 3: none of the format
    else:
        print('Failed to load a valid file')
        print('Returning an empty string')
        document = ''
    
    print(type(document))
    return document


# ### 3. Function to tokenize the document
def tokenize(document):
    # We are tokenizing using the PunktSentenceTokenizer
    # we call an instance of this class as sentence_tokenizer
    doc_tokenizer = PunktSentenceTokenizer()
    
    # tokenize() method: takes our document as input and returns a list of all the sentences in the document
    
    # sentences is a list containing each sentence of the document as an element
    sentences_list = doc_tokenizer.tokenize(document)
    return sentences_list


# ### 4. Read the document
document = readDoc()
print('The length of the file is:', end=' ')
print(len(document))


# ### 5. Generate a list of sentences in the document
sentences_list = tokenize(document)

print('The size of the list "sentences" is: {}'.format(len(sentences_list)))

for i in sentences_list:
    print(i)


# ### 6. Generate term-document matrix (TD matrix) of the data 

cv = CountVectorizer()
cv_matrix = cv.fit_transform(sentences_list)

#print(cv.get_feature_names())
print(cv_matrix.toarray())

normal_matrix = TfidfTransformer().fit_transform(cv_matrix)
print(normal_matrix.toarray())

print(normal_matrix.T.toarray)
res_graph = normal_matrix * normal_matrix.T
# plt.spy(res_graph)


# ### 7. Generate a graph for the document to apply PageRank algorithm  

# each node represents a sentence
# an edge represents that they have words in common
# the edge weight is the number of words that are common in both of the sentences(nodes)
# nx.draw() method is used to draw the graph created

nx_graph = nx.from_scipy_sparse_matrix(res_graph)
nx.draw_circular(nx_graph)
print('Number of edges {}'.format(nx_graph.number_of_edges()))
print('Number of vertices {}'.format(nx_graph.number_of_nodes()))
# plt.show()
print('The memory used by the graph in Bytes is: {}'.format(sys.getsizeof(nx_graph)))


# ### 8. Getting the rank of every sentence using pagerank

# ranks is a dictionary with key=node(sentences) and value=textrank (the rank of each of the sentences)
ranks = nx.pagerank(nx_graph)

print(type(ranks))
print('The size used by the dictionary in Bytes is: {}'.format(sys.getsizeof(ranks)))

for i in ranks:
    print(i, ranks[i])


# ### 9. Finding important sentences and generating summary

sentence_array = sorted(((ranks[i], s) for i, s in enumerate(sentences_list)), reverse=True)
sentence_array = np.asarray(sentence_array)

rank_max = float(sentence_array[0][0])
rank_min = float(sentence_array[len(sentence_array) - 1][0])

print(rank_max)
print(rank_min)

temp_array = []

flag = 0
if rank_max - rank_min == 0:
    temp_array.append(0)
    flag = 1

if flag != 1:
    for i in range(0, len(sentence_array)):
        temp_array.append((float(sentence_array[i][0]) - rank_min) / (rank_max - rank_min))

print(len(temp_array))

threshold = (sum(temp_array) / len(temp_array)) + 0.2


# Separate out the sentences that satisfy the criteria of having a score above the threshold
sentence_list = []
if len(temp_array) > 1:
    for i in range(0, len(temp_array)):
        if temp_array[i] > threshold:
                sentence_list.append(sentence_array[i][1])
else:
    sentence_list.append(sentence_array[0][1])


model = sentence_list


# ### 10. Writing the summary to a new file

summary = " ".join(str(x) for x in sentence_list)
print(summary)

f = open('sum.txt', 'a+')

f.write('-------------------\n')
f.write(summary)
f.write('\n')

f.close


for lines in sentence_list:
    print(lines)

