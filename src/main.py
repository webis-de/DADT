#!/usr/bin/python3
import sys, os, re
import numpy as np
from lda import *
from at import *
from dadt import *
import math
import copy

sys.path.append(os.path.abspath("liblinear/python"))
import liblinearutil as ll

DIRECTORY = '../data/nichtschiller'

print('Reading corpus')

docs_content = {}
doc_authors = {}
author_ids = {}  # id (int) → author name (string)
document_ids = {}

test_docs_content = {}
test_doc_authors = []
test_author_ids = {}  # id (int) → author name (string)
test_document_ids = {}

stopwords = []
stopfile = open("../data/germanST.txt")
for line in stopfile:
    line = line.strip()
    stopwords.append(line)
stopfile.close()


def train_test():
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
                author = match.group(1)
                if author not in author_ids.values():
                    author_id = next_author_id
                    author_ids[author_id] = author
                    next_author_id += 1
                else:
                    author_id = [key for key, value in author_ids.items() if value == author][0]

                with open(root + '/' + filename) as f:
                    content = []
                    for line in f:
                        words = [pattern.sub('', w.lower()) for w in line.split()]
                        content.extend(words)
                    test_document_ids[next_test_document_id] = filename
                    test_docs_content[next_test_document_id] = content
                    test_doc_authors.append(author_id)
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

    return(docs_content, test_docs_content)

def ten_fold_cross_validation():
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
                author = match.group(1)
                if author not in author_ids.values():
                    author_id = next_author_id
                    author_ids[author_id] = author
                    next_author_id += 1
                else:
                    author_id = [key for key, value in author_ids.items() if value == author][0]

                with open(root + '/' + filename) as f:
                    content = []
                    for line in f:
                        words = [pattern.sub('', w.lower()) for w in line.split()]
                        content.extend(words)
                    test_document_ids[next_test_document_id] = filename
                    test_docs_content[next_test_document_id] = content
                    test_doc_authors.append(author_id)
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

    return(docs_content, test_docs_content)



def preprocess(docs_content, test_docs_content):

    print('Building vocab')
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

    print('Building BOW representation')

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

    n_authors = len(author_ids)

    return(matrix, test_matrix, n_authors, vocab, test_doc_authors)

def add_fic_authors(doc_authors, n_authors):
    n_docs = len(doc_authors)
    next_fa = n_authors

    doc_authors_new = copy.deepcopy(doc_authors)

    for doc in doc_authors_new.keys():
        doc_authors_new[doc].append(next_fa)
        next_fa += 1

    return(doc_authors_new, next_fa)

def TOKEN_SVM(matrix, test_matrix):
    svm_model = ll.train(sum(doc_authors.values(), []), matrix.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(test_doc_authors, test_matrix.tolist(), svm_model)
    return p_label

def LDA_SVM(matrix, test_matrix):
    # set parameters
    number_topics = 20
    burn_in = 8  # 0
    alpha = 0.1
    beta = 0.1
    samples = 4
    spacing = 2  # 100
    chains = 2

    sampler = LDA(number_topics, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(matrix, burn_in, samples, spacing)
    print('likelihood: ', likelihood)

    theta_test, likelihood = sampler.classify(test_matrix, phi, burn_in, samples, spacing, chains)
    print('likelihood: ', likelihood)

    svm_model = ll.train(sum(doc_authors.values(), []), theta.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(test_doc_authors, theta_test.tolist(), svm_model)
    return p_label

def AT_SVM(matrix, test_matrix):
    # set parameters
    number_topics = 4
    burn_in = 8  # 0
    alpha = 0.1
    beta = 0.1
    samples = 2
    spacing = 2  # 100
    chains = 2

    sampler = AtSampler(number_topics, n_authors, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(doc_authors, matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    theta_test, likelihood = sampler.classify(test_matrix, phi, burn_in, samples, spacing, chains)
    print('theta: ', theta_test)
    print('likelihood: ', likelihood)

    svm_model = ll.train([0,1], theta.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(test_doc_authors, theta_test.tolist(), svm_model)
    print(p_label)

    for text in range(len(test_docs_content.items())):
        print(test_document_ids[text], author_ids[p_label[text]])

def AT_P(matrix, test_matrix, n_authors):
    # set parameters
    number_topics = 4
    burn_in = 5 # 0
    alpha = 0.1
    beta = 0.1
    samples = 1 # 0
    spacing = 5  # 100
    chains = 2

    sampler = AtSampler(number_topics, n_authors, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(doc_authors, matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    authors = sampler.at_p(phi, theta, test_matrix)

    return(authors)

def AT_FA_SVM(matrix, test_matrix):
    # set parameters
    number_topics = 4
    burn_in = 5  # 0
    alpha = 0.1
    beta = 0.1
    samples = 2
    spacing = 2  # 100
    chains = 2

    doc_authors_new, n_authors_new = add_fic_authors(doc_authors, n_authors)

    sampler = AtSampler(number_topics, n_authors_new, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(doc_authors_new, matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    sampler.n_authors = n_authors

    theta_test, likelihood = sampler.classify(test_matrix, phi, burn_in, samples, spacing, chains)
    print('theta: ', theta_test)
    print('likelihood: ', likelihood)

    num_training_docs = matrix.shape[0]
    training_matrix = np.zeros((num_training_docs, number_topics * 2))
    for doc in range(num_training_docs):
        training_doc_authors = doc_authors_new[doc]
        vector = np.concatenate(theta[training_doc_authors])
        training_matrix[doc] = vector

    num_test_docs = test_matrix.shape[0]
    test_matrix = np.concatenate((theta_test, theta_test), axis=1)

    svm_model = ll.train(sum(doc_authors.values(), []), training_matrix.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(test_doc_authors, test_matrix.tolist(), svm_model)

    return p_label

def AT_FA_P1(matrix, test_matrix):
    # set parameters
    number_topics = 4
    burn_in = 2 # 0
    alpha = 0.1
    beta = 0.1
    samples = 2
    spacing = 1  # 100
    chains = 2

    doc_authors_new, n_authors_new = add_fic_authors(doc_authors, n_authors)

    sampler = AtSampler(number_topics, n_authors_new, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(doc_authors_new, matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    sampler.n_authors = n_authors

    authors = sampler.at_p(phi, theta, test_matrix)

    return(authors)

def AT_FA_P2(matrix, test_matrix):
    # set parameters
    number_topics = 4
    burn_in = 5  # 0
    alpha = 0.1
    beta = 0.1
    samples = 2
    spacing = 2  # 100
    chains = 2

    doc_authors_new, n_authors_new = add_fic_authors(doc_authors, n_authors)

    sampler = AtSampler(number_topics, n_authors_new, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(doc_authors_new, matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    sampler.n_authors = n_authors

    authors = sampler.at_fa_p2(phi, theta, test_matrix, samples, burn_in, spacing)

    return(authors)

def DADT_SVM(matrix, test_matrix):
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
    test_samples = 2#10
    test_burn_in = 3#10
    test_spacing = 2#10
    chains = 1#2

    beta_a = np.array([0.01 + epsilon if word in stopwords else 0.01 for word in vocab])
    beta_d = np.array([0.01 - epsilon if word in stopwords else 0.01 for word in vocab])

    print('Starting!')
    (atopic_phi_sampled, atopic_theta_sampled, dtopic_phi_sampled, dtopic_theta_sampled, pi_sampled, chi) = train(matrix, vocab, doc_authors, num_dtopics, num_atopics, n_authors, alpha_a, beta_a, alpha_d, beta_d, eta, delta_a, delta_d, burn_in, samples, spacing)

    print("Testing")

    (dtopic_theta_test, atopic_theta_test, pi_test ) = classify(test_matrix, chains, test_burn_in, test_samples, test_spacing, num_dtopics, num_atopics, alpha_a, alpha_d, beta_a, beta_d, eta, delta_a, delta_d, dtopic_phi_sampled, atopic_phi_sampled)

#     training_matrix = np.concatenate((atopic_theta_sampled, dtopic_theta_sampled), axis=1)
#     test_matrix = np.concatenate((atopic_theta_test, dtopic_theta_test), axis=1)

    print("Main")
    num_training_docs = matrix.shape[0]
    num_test_docs = test_matrix.shape[0]
    training_matrix = np.zeros((num_training_docs, num_atopics + num_dtopics))
    for doc in range(num_training_docs):
        training_doc_author = doc_authors[doc][0]
        vector = np.concatenate((dtopic_theta_sampled[doc],atopic_theta_sampled[training_doc_author]))
        training_matrix[doc] = vector

    svm_test_matrix = np.zeros((num_test_docs, num_atopics + num_dtopics))
    for doc in range(num_test_docs):
        test_doc_author = doc_authors[doc][0]
        print(doc)
        print(test_doc_author)
        vector = np.concatenate((dtopic_theta_test[doc],atopic_theta_test[test_doc_author]))
        svm_test_matrix[doc] = vector

    svm_model = ll.train(sum(doc_authors.values(), []), training_matrix.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(test_doc_authors, svm_test_matrix.tolist(), svm_model)
    return p_label

def DADT_P(matrix, test_matrix, vocab):
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
    test_samples = 2
    test_burn_in = 3
    test_spacing = 2
    chains = 2

    beta_a = np.array([0.01 + epsilon if word in stopwords else 0.01 for word in vocab])
    beta_d = np.array([0.01 - epsilon if word in stopwords else 0.01 for word in vocab])

    print('Starting!')
    (atopic_phi_sampled, atopic_theta_sampled, dtopic_phi_sampled, dtopic_theta_sampled, pi_sampled, chi) = train(matrix, vocab, doc_authors, num_dtopics, num_atopics, n_authors, alpha_a, beta_a, alpha_d, beta_d, eta, delta_a, delta_d, burn_in, samples, spacing)

    print("Classifying")

    (atopic_theta_test, dtopic_theta_test, pi_test ) = classify(matrix, chains, test_burn_in, test_samples, test_spacing, num_dtopics, num_atopics, alpha_a, alpha_d, beta_a, beta_d, eta, delta_a, delta_d, dtopic_phi_sampled, atopic_phi_sampled)

    print("Deciding")

    authors = dadt_p(test_matrix, n_authors, atopic_phi_sampled, dtopic_phi_sampled, atopic_theta_sampled, dtopic_theta_test, pi_test, chi)

    return(authors)

def run_model(matrix, test_matrix, n_authors, vocab):
    # author_guess = TOKEN_SVM(matrix, test_matrix)
    # author_guess = LDA_SVM(matrix, test_matrix)
    # author_guess = AT_SVM(matrix, test_matrix)
    # author_guess = AT_P(matrix, test_matrix, n_authors)
    # author_guess = AT_FA_SVM(matrix, test_matrix)
    # author_guess = AT_FA_P1(matrix, test_matrix)
    # author_guess = AT_FA_P2(matrix, test_matrix)
    author_guess = DADT_SVM(matrix, test_matrix)
    # author_guess = DADT_P(matrix, test_matrix, vocab)

    return author_guess

# (docs_content, test_docs_content) = train_test()
(docs_content, test_docs_content) = ten_fold_cross_validation()
(matrix, test_matrix, n_authors, vocab, test_doc_authors) = preprocess(docs_content, test_docs_content)

author_guess = run_model(matrix, test_matrix, n_authors, vocab)

for text in range(len(test_docs_content.items())):
    print(test_document_ids[text], author_ids[author_guess[text]])

accuracy=np.equal(test_doc_authors,author_guess).mean()

print(accuracy)
