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

stopwords = []
stopfile = open("../data/germanST.txt")
for line in stopfile:
    line = line.strip()
    stopwords.append(line)
stopfile.close()

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
    samples = 4
    spacing = 2  # 100
    chains = 2

    sampler = LdaSampler(number_topics, alpha, beta)

    info('Starting!')
    theta, phi, likelihood = sampler.train(matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    theta, likelihood = sampler.classify(test_matrix, phi, burn_in, samples, spacing, chains)
    print('theta: ', theta)
    print('likelihood: ', likelihood)

def runAT():
    # set parameters
    number_topics = 4
    burn_in = 8  # 0
    alpha = 0.1
    beta = 0.1
    samples = 4
    spacing = 2  # 100
    chains = 2

    sampler = AtSampler(number_topics, len(author_ids), alpha, beta)

    info('Starting!')
    theta, phi, likelihood = sampler.train(doc_authors, matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    theta, likelihood = sampler.classify(test_matrix, phi, burn_in, samples, spacing, chains)
    print('theta: ', theta)
    print('likelihood: ', likelihood)


def runDADT():
    # set parameters
    num_atopics = 15
    num_dtopics = 5
    burn_in = 1  # 1000
    alpha_a = min(0.1, 5/num_atopics)
    alpha_d = min(0.1, 5/num_dtopics)
    eta = 1
    epsilon = 0.009
    delta_a = 4.889
    delta_d = 1.222
    samples = 2
    spacing = 1 # 100
    test_samples = 10
    test_burn_in = 10
    test_spacing = 10
    chains = 2

    beta_a = np.array([0.01 + epsilon if word in stopwords else 0.01 for word in vocab])
    beta_d = np.array([0.01 - epsilon if word in stopwords else 0.01 for word in vocab])

    print('Starting!')
    (atopic_phi_sampled, atopic_theta_sampled, dtopic_phi_sampled, dtopic_theta_sampled, pi_sampled, chi) = train(matrix, vocab, doc_authors, num_dtopics, num_atopics, len(author_ids), alpha_a, beta_a, alpha_d, beta_d, eta, delta_a, delta_d, burn_in, samples, spacing)

    print("Testing")

    (dtopic_theta_test, atopic_theta_test, pi_test ) = classify(matrix, chains, test_burn_in, test_samples, test_spacing, num_dtopics, num_atopics, alpha_a, alpha_d, beta_a, beta_d, eta, delta_a, delta_d, dtopic_phi_sampled, atopic_phi_sampled)

# runLDA()
# runAT()
runDADT()
