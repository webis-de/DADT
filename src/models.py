#!/usr/bin/python3
import numpy as np
import random
import lda
import at
import dadt
import copy
import sys, os
sys.path.append(os.path.abspath("liblinear/python"))
import liblinearutil as ll

a = "a"
d = "d"

def concatenate_fic_authors(doc_authors, num_topics):
    num_training_docs = len(doc_authors)
    training_matrix = np.zeros((num_training_docs, num_topics * 2))
    for doc, doc_author_two in doc_authors.items():
        vector = np.concatenate(theta[doc_author_two])
        training_matrix[doc] = vector

    return training_matrix

def add_fic_authors(doc_authors, n_authors):
    n_docs = len(doc_authors)
    next_fa = n_authors

    doc_authors_new = copy.deepcopy(doc_authors)

    for doc, list in enumerate(doc_authors_new):
        doc_authors_new[doc].append(next_fa)
        next_fa += 1

    return(doc_authors_new, next_fa)

def TOKEN_SVM(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords):
    n_docs = matrix.shape[0]
    n_test_docs = test_matrix.shape[0]
    matrix = matrix / np.sum(matrix, 1)[:,None]
    test_matrix = test_matrix / np.sum(test_matrix, 1)[:,None]

    svm_model = ll.train(sum(doc_authors, []), matrix.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(np.random.rand(n_test_docs), test_matrix.tolist(), svm_model)

    author_probs = np.zeros((n_test_docs, n_authors))
    for doc, author in enumerate(p_label):
        author_probs[doc,int(author)] = 1

    return author_probs

def LDA_SVM(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords):
    # set parameters
    num_topics = 20
    burn_in = 1000  # 0
    alpha = 0.1
    beta = 0.1
    samples = 8
    spacing = 100

    num_test_docs = test_matrix.shape[0]

    sampler = lda.LDA(num_topics, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(matrix, burn_in, samples, spacing)
    print('likelihood: ', likelihood)

    theta_test, likelihood = sampler.classify(test_matrix, phi, burn_in, samples, spacing)
    print('likelihood: ', likelihood)


    theta = theta / np.sum(theta, 1)[:,None]
    theta_test = theta_test / np.sum(theta_test, 1)[:,None]

    svm_model = ll.train(sum(doc_authors, []), theta.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(np.random.rand(num_test_docs), theta_test.tolist(), svm_model)
    author_probs = np.zeros((n_test_docs, n_authors))
    for doc, author in enumerate(p_label):
        author_probs[doc,int(author)] = 1

    return author_probs


def AT_SVM(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords):
    # set parameters
    num_topics = 4
    burn_in = 1000  # 0
    alpha = 0.1
    beta = 0.1
    samples = 8
    spacing = 100

    num_test_docs = test_matrix.shape[0]

    sampler = at.AtSampler(num_topics, n_authors, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(doc_authors, matrix, burn_in, samples, spacing)
    print('theta: ', theta.shape)
    print('phi: ', phi.shape)
    print('likelihood: ', likelihood)

    sampler.n_authors = num_test_docs

    theta_test = sampler.classify(test_matrix, phi, burn_in, samples, spacing)
    print('theta test: ', theta_test.shape)
    print('likelihood: ', likelihood)

    theta = theta / np.sum(theta, 1)[:,None]
    theta_test = theta_test / np.sum(theta_test, 1)[:,None]

    svm_model = ll.train([0,1], theta.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(np.random.rand(num_test_docs), theta_test.tolist(), svm_model)
    author_probs = np.zeros((n_test_docs, n_authors))
    for doc, author in enumerate(p_label):
        author_probs[doc,int(author)] = 1

    return author_probs

def AT_P(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords):
    # set parameters
    num_topics = 400
    burn_in = 1000
    alpha = 0.1
    beta = 0.1
    samples = 8
    spacing = 100

    sampler = at.AtSampler(num_topics, n_authors, alpha, beta)

    print('Starting!')
    theta, phi = sampler.train(doc_authors, matrix, burn_in, samples, spacing)
    print('theta: ', theta.shape)
    print('phi: ', phi.shape)

    author_probs = sampler.at_p(phi, theta, test_matrix)

    return author_probs

def AT_FA_SVM(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords):
    # set parameters
    num_topics = 4
    burn_in = 1000  # 0
    alpha = 0.1
    beta = 0.1
    samples = 8
    spacing = 100

    num_test_docs = test_matrix.shape[0]

    doc_authors_new, n_authors_new = add_fic_authors(doc_authors, n_authors)

    sampler = at.AtSampler(num_topics, n_authors_new, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(doc_authors_new, matrix, burn_in, samples, spacing)
    print('theta:', theta.shape)
    print('phi:', phi.shape)
    print('likelihood:', likelihood)

    sampler.n_authors = num_test_docs

    theta_test = sampler.classify(test_matrix, phi, burn_in, samples, spacing)
    print('theta test:', theta_test.shape)

    training_matrix = concatenate_fic_authors(doc_authors, num_topics)

    num_test_docs = test_matrix.shape[0]
    test_matrix = np.concatenate((theta_test, theta_test), axis=1)

    training_matrix = training_matrix / np.sum(training_matrix, 1)[:,None]
    test_matrix = test_matrix / np.sum(test_matrix, 1)[:,None]

    svm_model = ll.train(sum(doc_authors, []), training_matrix.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(np.random.rand(num_test_docs), test_matrix.tolist(), svm_model)

    author_probs = np.zeros((n_test_docs, n_authors))
    for doc, author in enumerate(p_label):
        author_probs[doc,int(author)] = 1

    return author_probs

def AT_FA_P1(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords):
    # set parameters
    num_topics = 4
    burn_in = 1000 # 0
    alpha = 0.1
    beta = 0.1
    samples = 8
    spacing = 100

    doc_authors_new, n_authors_new = add_fic_authors(doc_authors, n_authors)

    sampler = at.AtSampler(num_topics, n_authors_new, alpha, beta)

    print('Starting!')
    theta, phi= sampler.train(doc_authors_new, matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)

    sampler.n_authors = n_authors

    author_probs = sampler.at_p(phi, theta, test_matrix)

    return(author_probs)

def AT_FA_P2(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords):
    # set parameters
    num_topics = 4
    burn_in = 1000  # 0
    alpha = 0.1
    beta = 0.1
    samples = 8
    spacing = 100

    doc_authors_new, n_authors_new = add_fic_authors(doc_authors, n_authors)

    sampler = at.AtSampler(num_topics, n_authors_new, alpha, beta)

    print('Starting!')
    theta, phi, likelihood = sampler.train(doc_authors_new, matrix, burn_in, samples, spacing)
    print('theta: ', theta)
    print('phi: ', phi)
    print('likelihood: ', likelihood)

    sampler.n_authors = n_authors

    authors = sampler.at_fa_p2(phi, theta, test_matrix, samples, burn_in, spacing)

    return(author_probs)

def DADT_SVM(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords):
    # set parameters
    num_atopics = 15
    num_dtopics = 5
    burn_in = 1000  # 1000
    alpha_a = min(0.1, 5/num_atopics)
    alpha_d = min(0.1, 5/num_dtopics)
    eta = 1
    epsilon = 0.009
    delta_a = 4.889
    delta_d = 1.222
    samples = 8
    spacing = 100
    test_samples = 10
    test_burn_in = 10
    test_spacing = 1

    num_test_docs = test_matrix.shape[0]

    beta_a = np.array([0.01 + epsilon if word in stopwords else 0.01 for word in vocab])
    beta_d = np.array([0.01 - epsilon if word in stopwords else 0.01 for word in vocab])

    print('Starting!')
    (atopic_phi_sampled, atopic_theta_sampled, dtopic_phi_sampled, dtopic_theta_sampled, pi_sampled, chi) = dadt.train(matrix, vocab, doc_authors, num_dtopics, num_atopics, n_authors, alpha_a, beta_a, alpha_d, beta_d, eta, delta_a, delta_d, burn_in, samples, spacing)

    print("Testing")

    (dtopic_theta_test, atopic_theta_test, pi_test ) = dadt.classify(test_matrix, test_burn_in, test_samples, test_spacing, num_dtopics, num_atopics, alpha_a, alpha_d, beta_a, beta_d, eta, delta_a, delta_d, dtopic_phi_sampled, atopic_phi_sampled)

    print("Main")

    num_test_docs = test_matrix.shape[0]
    training_matrix = concatenate_fic_authors(doc_authors, num_topics)

    svm_test_matrix = np.zeros((num_test_docs, num_atopics + num_dtopics))

    for doc in range(num_test_docs):
        vector = np.concatenate((dtopic_theta_test[doc],atopic_theta_test[doc]))
        svm_test_matrix[doc] = vector

    training_matrix = training_matrix / np.sum(training_matrix, 1)[:,None]
    svm_test_matrix = svm_test_matrix / np.sum(svm_test_matrix, 1)[:,None]

    svm_model = ll.train(sum(doc_authors, []), training_matrix.tolist(), '-c 4')
    p_label, p_acc, p_val = ll.predict(np.random.rand(num_test_docs), svm_test_matrix.tolist(), svm_model)

    author_probs = np.zeros((n_test_docs, n_authors))
    for doc, author in enumerate(p_label):
        author_probs[doc,int(author)] = 1

    return author_probs


def DADT_P(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords):
    # set parameters
    num_dtopics = 50
    num_atopics = 350
    burn_in = 1000
    alpha_a = min(0.1, 5/num_atopics)
    alpha_d = min(0.1, 5/num_dtopics)
    eta = 1
    epsilon = 0.009
    delta_a = 4.889
    delta_d = 1.222
    samples = 8
    spacing = 100
    test_samples = 10
    test_burn_in = 10
    test_spacing = 1

    beta_a = np.array([0.01 + epsilon if word in stopwords else 0.01 for word in vocab])
    beta_d = np.array([0.01 - epsilon if word in stopwords else 0.01 for word in vocab])

    alpha = {a: alpha_a, d: alpha_d}
    beta = {a: beta_a, d: beta_d}
    delta = {a: alpha_a, d: alpha_d}
    num_topics = {a: num_atopics, d: num_dtopics}

    print('Starting!')
    (theta_sampled, phi_sampled, pi_sampled, chi_sampled) = dadt.train(matrix, vocab, doc_authors, num_topics, n_authors, alpha, beta, delta, eta, burn_in, samples, spacing)

    print("Classifying")

    (theta_test, pi_test) = dadt.classify(test_matrix, test_burn_in, test_samples, test_spacing, num_topics, alpha, beta, delta, eta, phi_sampled)

    print("Deciding")

    author_probs = dadt.dadt_p(test_matrix, n_authors, theta_sampled, phi_sampled, pi_test, chi_sampled)

    return(author_probs)
