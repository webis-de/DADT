#!/usr/bin/python3
import sys, os, re
from termcolor import colored
import numpy as np
from lda import *
from at import *
from dadt import * 
import math


def info(s):
    print(colored(s, 'yellow'))


DIRECTORY = '../data/nichtschiller'

info('Reading corpus')

docs_content = {}
doc_authors = {}
author_ids = {}  # id (int) → author name (string)
document_ids = {}

test_docs_content = {}
test_doc_authors = {}
test_author_ids = {}  # id (int) → author name (string)
test_document_ids = {}

pattern = re.compile(r'[\W_]+')
author_pattern = re.compile(r'([a-zäöüA-ZÄÖÜ\-]+)\d+\.txt')

next_author_id = 0
next_document_id = 0
next_test_document_id = 0

for root, dirs, files in os.walk(DIRECTORY):
    isTest = not (len(dirs) and dirs[-1] == "test")
    for filename in files:
        match = author_pattern.match(filename)
        if not match:
            print('No match')
            continue

        if isTest:
            with open(root + '/' + filename) as f:
                content = []
                for line in f:
                    words = [pattern.sub('', w.lower()) for w in line.split()]
                    content.extend(words)
                test_document_ids[next_test_document_id] = filename
                test_docs_content[next_test_document_id] = content
                next_test_document_id += 1

        else:
            authors = match.group(1).split("-")
            doc_author_ids = []
            for author in authors:
                if author not in author_ids.values():
                    author_id = next_author_id
                    author_ids[author_id] = author
                    next_author_id += 1
                else:
                    author_id = [key for key, value in author_ids.items() if value == author][0]
                doc_author_ids.append(author_id)

            with open(root + '/' + filename) as f:
                content = []
                for line in f:
                    words = [pattern.sub('', w.lower()) for w in line.split()]
                    content.extend(words)
                document_ids[next_document_id] = filename
                docs_content[next_document_id] = content
                doc_authors[next_document_id] = doc_author_ids
                next_document_id += 1

info('Building vocab')
vocab = set()
for doc, content in docs_content.items():
    for word in content:
        if len(word) > 1:
            vocab.add(word)

for doc, content in test_docs_content.items():
    for word in content:
        if len(word) > 1:
            vocab.add(word)

vocab = list(vocab)
lookupvocab = dict([(word, index) for (index, word) in enumerate(vocab)])

info('Building BOW representation')
matrix = np.zeros((len(docs_content), len(vocab)))

test_matrix = np.zeros((len(test_docs_content), len(vocab)))

for doc, content in docs_content.items():
    for word in content:
        if len(word) > 1:
            matrix[doc, lookupvocab[word]] += 1

for doc, content in test_docs_content.items():
    for word in content:
        if len(word) > 1:
            test_matrix[doc, lookupvocab[word]] += 1

def runLDA():
    # set parameters
    number_topics = 20
    burn_in = 8  # 0
    alpha = 0.1
    beta = 0.1
    samples = 8
    spacing = 2  # 100

    sampler = LdaSampler(number_topics, alpha, beta)

    info('Starting!')
    theta, phi, likelihood = sampler.train(matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    # theta = sampler.test(test_matrix, phi, burn_in, samples, spacing)


def runAT():
    # set parameters
    number_topics = 2
    burn_in = 50  # 0
    alpha = 0.1
    beta = 0.1
    samples = 5
    spacing = 10  # 100

    sampler = AtSampler(number_topics, len(author_ids), alpha, beta)

    info('Starting!')
    theta, phi, likelihood = sampler.run(doc_authors, matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    for topic in range(number_topics):
        words = {i : phi[topic, i] for i in range(len(vocab))}
        sorted_words = sorted(words, key=words.get)
        print('Topic ', topic)
        for i in range(10):
            print(vocab[sorted_words[i]])

def runDADT():
    # set parameters
    number_topics = 2
    burn_in = 50  # 0
    alpha = 0.1
    beta = 0.1
    delta_A = 0.5
    delta_D = 0.5
    samples = 5
    spacing = 10  # 100


    info('Starting!')
    # def learn(matrix, doc_authors, num_dtopics, num_atopics, num_authors, alpha_a, beta_a, alpha_d, beta_d, eta, delta_A, delta_D, burn_in, samples, spacing):
    theta, phi, likelihood = learn(matrix, doc_authors, 10, 10, len(author_ids), alpha, beta, alpha, beta, 0, delta_A, delta_D, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    # for topic in range(number_topics):
    #     words = {i : phi[topic, i] for i in range(len(vocab))}
    #     sorted_words = sorted(words, key=words.get)
    #     print('Topic ', topic)
    #     for i in range(10):
    #         print(vocab[sorted_words[i]])



# runLDA()
# runAT()
runDADT()
