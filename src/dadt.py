#!/usr/bin/python3
import numpy as np
from termcolor import colored
def info(s):
    print(colored(s, 'blue'))

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:  # if word 5 appears 3 times, returns 5 5 5
        for i in range(int(vec[idx])):
            yield idx


def dtopic_distribution(cooccurrenceence_dtopic_word, cooccurrenceence_doc_dtopic, alpha, beta):
    phi = cooccurrenceence_dtopic_word + beta
    phi /= np.sum(phi, axis=1)[:, np.newaxis]

    theta = cooccurrenceence_doc_dtopic + alpha
    theta /= np.sum(theta, axis=1)[:, np.newaxis]

    return phi, theta

def atopic_distribution(cooccurrenceence_atopic_word, cooccurrenceence_author_atopic, alpha, beta):
    # word topic distribution
    phi = cooccurrenceence_atopic_word + beta
    phi /= np.sum(phi, axis=1)[:, np.newaxis]

    # author topic distribution
    theta = cooccurrenceence_author_atopic + alpha
    theta /= np.sum(theta, axis=1)[:, np.newaxis]

    return phi, theta

def dtopic_phi(cooccurrence_dtopic_word, beta):
    phi = (cooccurrence_dtopic_word + beta)
    phi /= np.sum(phi, axis=1)[:, np.newaxis]
    return phi

def dtopic_theta(cooccurrence_doc_dtopic, alpha):
    theta = cooccurrence_doc_dtopic + alpha
    theta /= np.sum(theta, axis=1)[:, np.newaxis]
    return theta

def sample_topic(doc, word, num_dtopics, doc_authors, cooccurrence_atopic_word, occurrence_atopic, cooccurrence_author_atopic, occurrence_author, num_atopics, num_awords_per_doc, cooccurrence_dtopic_word, occurrence_dtopic, cooccurrence_doc_dtopic, num_dwords_per_doc, alpha_a, beta_a, delta_a, alpha_d, beta_d, delta_d):
    authors = doc_authors[doc]

    vocab_size = cooccurrence_dtopic_word.shape[1]

    document_dtopics = (cooccurrence_doc_dtopic[doc, :] + alpha_d) / \
                      (num_dwords_per_doc[doc] + alpha_d * num_dtopics)

    word_dtopics = (cooccurrence_dtopic_word[:, word] + beta_d[word]) / \
                  (occurrence_dtopic + beta_d[word] * vocab_size)

    distribution_d = (delta_d + num_dwords_per_doc[doc]) * document_dtopics * word_dtopics

    # normalize to obtain probabilities
    distribution_d /= np.sum(distribution_d)

    while np.sum(distribution_d) > 1.0:
        distribution_d = distribution_d * 0.999999999999999 #dirty hack because of floating point errors

    vocab_size = cooccurrence_atopic_word.shape[1]

    author_atopics = (cooccurrence_author_atopic[authors, :] + alpha_a) / (occurrence_author[authors].repeat(num_atopics).reshape(len(authors), num_atopics) + alpha_a * num_atopics)

    word_atopics = (cooccurrence_atopic_word[:, word] + beta_d[word]) / (occurrence_atopic + beta_d[word] * vocab_size)

    distribution_a = (delta_a + num_awords_per_doc[doc]) * author_atopics * word_atopics

    # reshape into a looong vector
    distribution_a = distribution_a.reshape(len(authors) * num_atopics)

    # normalize to obtain probabilities
    distribution_a /= np.sum(distribution_a)

    while np.sum(distribution_a) > 1.0:
        distribution_a = distribution_a * 0.999999999999999 #dirty hack because of floating point errors

    distribution = np.concatenate((distribution_d, distribution_a))
    distribution /= np.sum(distribution)
    idx = np.random.multinomial(1, distribution).argmax()

    is_dtopic = idx < num_dtopics

    if is_dtopic:
        return (not is_dtopic, 0, idx)
    else:
        idx -= num_dtopics
        new_author = doc_authors[doc][int(idx / num_atopics)]
        new_atopic = idx % num_atopics
        return (not is_dtopic, new_author, new_atopic)


def atopic_phi(cooccurrence_atopic_word, beta):
    phi = (cooccurrence_atopic_word + beta)
    phi /= np.sum(phi, axis=1)[:, np.newaxis]
    return phi

def atopic_theta(cooccurrence_author_atopic, alpha):
    theta = cooccurrence_author_atopic + alpha
    theta /= np.sum(theta, axis=1)[:, np.newaxis]
    return theta

def learn(matrix, vocab, doc_authors, num_dtopics, num_atopics, num_authors, alpha_a, beta_a, alpha_d, beta_d, eta, delta_a, delta_d, burn_in, samples, spacing):
    num_docs, num_words = matrix.shape

    # Initialize matricies
    cooccurrence_dtopic_word = np.zeros((num_dtopics, num_words))
    cooccurrence_atopic_word = np.zeros((num_atopics, num_words))
    cooccurrence_author_atopic = np.zeros((num_authors, num_atopics))
    prior_author = np.zeros((num_authors))  # only used in classifier
    cooccurrence_doc_dtopic = np.zeros((num_docs, num_dtopics))

    occurrence_atopic = np.zeros((num_atopics))
    occurrence_dtopic = np.zeros((num_dtopics))
    occurrence_author = np.zeros((num_authors))

    num_dwords_per_doc = np.zeros((num_docs))
    num_awords_per_doc = np.zeros((num_docs))

    document_atopic_dtopic_ratio = np.zeros((num_docs))
    document_word_topic = {}  # index : (doc, word-index) value: (1 for atopic 0 for dtopic, topic)
    authors = {}

    for doc in range(num_docs):
        document_atopic_dtopic_ratio[doc] = np.random.beta(delta_a, delta_d)
        # i is a number between 0 and doc_length-1
        # w is a number between 0 and vocab_size-1
        for i, word in enumerate(word_indices(matrix[doc, :])):
            # choose an arbitrary topic as first topic for word i
            is_atopic = np.random.binomial(1, document_atopic_dtopic_ratio[doc])
            if (is_atopic):
                atopic = np.random.randint(num_atopics)
                author = doc_authors[doc][np.random.randint(len(doc_authors[doc]))]
                occurrence_atopic[atopic] += 1
                occurrence_author[author] += 1
                document_word_topic[(doc, i)] = (is_atopic, atopic)
                cooccurrence_atopic_word[(atopic, word)] += 1
                cooccurrence_author_atopic[(author, atopic)] += 1
                authors[(doc, i)] = author
                num_awords_per_doc[doc] += 1
            else:
                dtopic = np.random.randint(num_dtopics)
                cooccurrence_dtopic_word[(dtopic, word)] += 1
                cooccurrence_doc_dtopic[(doc, dtopic)] += 1
                occurrence_dtopic[dtopic] += 1
                num_dwords_per_doc[doc] += 1
                document_word_topic[(doc, i)] = (is_atopic, dtopic)

    taken_samples = 0

    it = 0  # iterations
    atopic_theta_sampled = 0;
    atopic_phi_sampled = 0;
    dtopic_theta_sampled = 0;
    dtopic_phi_sampled = 0;
    while taken_samples < samples:
        # print('Iteration ', it)
        for doc in range(num_docs):  # all documents
            for i, word in enumerate(word_indices(matrix[doc, :])):

                old_is_atopic, old_topic = document_word_topic[(doc, i)]

                if (old_is_atopic):
                    old_author = authors[(doc, i)]
                    cooccurrence_atopic_word[(old_topic, word)] -= 1
                    cooccurrence_author_atopic[(old_author, old_topic)] -= 1
                    occurrence_atopic[old_topic] -= 1
                    occurrence_author[old_author] -= 1
                    num_awords_per_doc[doc] -= 1
                else:
                    cooccurrence_dtopic_word[(old_topic, word)] -= 1
                    cooccurrence_doc_dtopic[(doc, old_topic)] -= 1
                    occurrence_dtopic[old_topic] -= 1
                    num_dwords_per_doc[doc] -= 1

                is_atopic, new_author, new_topic = sample_topic(doc, word, num_dtopics, doc_authors, cooccurrence_atopic_word, occurrence_atopic, cooccurrence_author_atopic, occurrence_author, num_atopics, num_awords_per_doc, cooccurrence_dtopic_word, occurrence_dtopic, cooccurrence_doc_dtopic, num_dwords_per_doc, alpha_a, beta_a, delta_a, alpha_d, beta_d, delta_d)

                if (is_atopic):
                    authors[(doc, i)] = new_author
                    cooccurrence_atopic_word[(new_topic, word)] += 1
                    cooccurrence_author_atopic[(new_author, new_topic)] += 1
                    occurrence_atopic[new_topic] += 1
                    occurrence_author[new_author] += 1
                    num_awords_per_doc[doc] += 1
                    document_word_topic[(doc, i)] = (is_atopic, new_topic)
                else:
                    cooccurrence_dtopic_word[(new_topic, word)] += 1
                    cooccurrence_doc_dtopic[(doc, new_topic)] += 1
                    occurrence_dtopic[new_topic] += 1
                    num_dwords_per_doc[doc] += 1
                    document_word_topic[(doc, i)] = (is_atopic, new_topic)

        if it >= burn_in:
            it_after_burn_in = it - burn_in
            if (it_after_burn_in % spacing) == 0:
                print('    Sampling!')
                atopic_phi_sampled += atopic_phi(cooccurrence_atopic_word, beta_a)
                atopic_theta_sampled += atopic_theta(cooccurrence_author_atopic, alpha_a)
                dtopic_phi_sampled += dtopic_phi(cooccurrence_dtopic_word, beta_d)
                dtopic_theta_sampled += dtopic_theta(cooccurrence_doc_dtopic, alpha_d)
                taken_samples += 1

        info ("\n\nIteration %s" % it)
        # print ("Likelihood", sampler.loglikelihood())
        print("\nAuthor topics")
        for topicNum in range(num_atopics):
            print("Topic",topicNum,":\t",end="")
            words = [(proba, w) for (w, proba) in enumerate(atopic_phi(cooccurrence_atopic_word, beta_a)[topicNum, :]) if proba > 0]
            words = sorted(words, reverse = True)
            for i in range(10):
                proba, w = words[i]
                print(vocab[w], end= " ")
            print(" ")

        print("\nDocument topics")
        for topicNum in range(num_dtopics):
            print("Topic",topicNum,":\t",end="")
            words = [(proba, w) for (w, proba) in enumerate(dtopic_phi(cooccurrence_dtopic_word, beta_d)[topicNum, :]) if proba > 0]
            words = sorted(words, reverse = True)
            for i in range(10):
                proba, w = words[i]
                print(vocab[w], end= " ")
            print(" ")

        it += 1

    atopic_phi_sampled /= taken_samples
    atopic_theta_sampled /= taken_samples
    dtopic_phi_sampled /= taken_samples
    dtopic_theta_sampled /= taken_samples

    return(atopic_phi_sampled, atopic_theta_sampled, dtopic_phi_sampled, dtopic_theta_sampled)


def classify(matrix, burn_in, samples, spacing, alpha_a, alpha_d, beta_a, beta_d, eta, delta_a, delta_d):
    print("error: not yet implemented")
    return [0]
