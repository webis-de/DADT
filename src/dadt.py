#!/usr/bin/python3
import numpy as np

a = "a"
d = "d"

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for index in vec.nonzero()[0]:  # if word 5 appears 3 times, returns 5 5 5
        for i in range(int(vec[index])):
            yield index

def topic_phi(a_d, cooccurrence_topic_word, beta):
    phi = (cooccurrence_topic_word[a_d] + beta[a_d])
    phi /= np.sum(phi, axis=1)[:, np.newaxis]
    return phi

def topic_theta(a_d, cooccurrence_authordoc_topic, alpha):
    theta = cooccurrence_authordoc_topic[a_d] + alpha[a_d]
    theta /= np.sum(theta, axis=1)[:, np.newaxis]
    return theta

def pi(delta, n_words_per_doc):
    pi = delta[a] + n_words_per_doc[a]
    pi /= (delta[d] + delta[a] + (n_words_per_doc[d] + n_words_per_doc[a]))
    return pi

def chi(eta, doc_authors,n_authors):
    n_docs_per_author = np.zeros((n_authors))
    for author in doc_authors:
        n_docs_per_author[author] += 1

    chi = eta + n_docs_per_author
    chi /= np.sum(chi)
    return chi

def normalise(distribution):

    distribution /= np.sum(distribution)

    while True:
        try:
            _ = np.random.multinomial(1, distribution).argmax()
            break
        except ValueError:
            distribution = distribution * 0.999999999999999  # dirty hack because of floating point errors

    return distribution

def sample_topic(doc, word, n_topics, doc_authors, cooccurrence_topic_word, occurrence_topic, cooccurrence_authordoc_topic, occurrence_author, n_words_per_doc, alpha, beta, delta):

    authors = doc_authors[doc]

    vocab_size_d = cooccurrence_topic_word[d].shape[1]

    document_dtopics = (cooccurrence_authordoc_topic[d][doc, :] + alpha[d]) / (n_words_per_doc[d][doc] + alpha[d] * n_topics[d])

    word_dtopics = (cooccurrence_topic_word[d][:, word] + beta[d][word]) / (occurrence_topic[d] + beta[d][word] * vocab_size_d)

    distribution_d = (delta[d] + n_words_per_doc[d][doc]) * document_dtopics * word_dtopics

    distribution_d = normalise(distribution_d)

    ###############################################################################################################################

    vocab_size_a = cooccurrence_topic_word[a].shape[1]

    author_atopics = cooccurrence_authordoc_topic[a][authors, :] + alpha[a]
    author_atopics /= np.sum(author_atopics, axis=1)[:, np.newaxis]

    word_atopics = (cooccurrence_topic_word[a] + beta[a])
    word_atopics /= np.sum(word_atopics, axis=1)[:, np.newaxis]
    word_atopics = word_atopics[:, word]

    distribution_a = (delta[a] + n_words_per_doc[a][doc]) * author_atopics * word_atopics
    distribution_a = distribution_a.reshape(len(authors) * n_topics[a])
    distribution_a = normalise(distribution_a)

    #########################################################################################################################

    distribution = np.concatenate((distribution_d, distribution_a))
    distribution /= np.sum(distribution)
    index = np.random.multinomial(1, distribution).argmax()

    is_dtopic = index < n_topics[d]

    if is_dtopic:
        return (not is_dtopic, 0, index)
    else:
        index -= n_topics[d]
        new_author = doc_authors[doc][int(index / n_topics[a])]
        new_atopic = index % n_topics[a]
        return (not is_dtopic, new_author, new_atopic)

def train(matrix, vocab, doc_authors, n_topics, n_authors, alpha, beta, delta, eta, burn_in, samples, spacing):
    n_docs, n_words = matrix.shape
    chi_sampled = chi(eta, doc_authors,n_authors)
    # Initialize matricies

    n_words_per_doc = {a : np.zeros((n_docs)), d : np.zeros((n_docs))}
    occurrence_author = np.zeros((n_authors))
    occurrence_topic = {a : np.zeros((n_topics[a])), d : np.zeros((n_topics[d]))}
    cooccurrence_authordoc_topic = {a: np.zeros((n_authors, n_topics[a])), d: np.zeros((n_docs, n_topics[d]))}
    document_atopic_dtopic_ratio = np.zeros((n_docs))
    document_word_topic = {}  # index : (doc, word-index) value: (1 for atopic 0 for dtopic, topic)

    cooccurrence_topic_word = {a : np.zeros((n_topics[a], n_words)), d : np.zeros((n_topics[d], n_words))}
    prior_author = np.zeros((n_authors))  # only used in classifier
    authors = {}

    for doc in range(n_docs):
        document_atopic_dtopic_ratio[doc] = np.random.beta(delta[a], delta[d])
        # i is a number between 0 and doc_length-1
        # w is a number between 0 and vocab_size-1
        for i, word in enumerate(word_indices(matrix[doc, :])):
            # choose an arbitrary topic as first topic for word i
            is_atopic = np.random.binomial(1, document_atopic_dtopic_ratio[doc])
            if (is_atopic):
                atopic = np.random.randint(n_topics[a])
                author = doc_authors[doc][np.random.randint(len(doc_authors[doc]))]
                occurrence_topic[a][atopic] += 1
                occurrence_author[author] += 1
                document_word_topic[(doc, i)] = (is_atopic, atopic)
                cooccurrence_topic_word[a][atopic, word] += 1
                cooccurrence_authordoc_topic[a][(author, atopic)] += 1
                authors[(doc, i)] = author
                n_words_per_doc[a][doc] += 1
            else:
                dtopic = np.random.randint(n_topics[d])
                cooccurrence_topic_word[d][(dtopic, word)] += 1
                cooccurrence_authordoc_topic[d][(doc, dtopic)] += 1
                occurrence_topic[d][dtopic] += 1
                n_words_per_doc[d][doc] += 1
                document_word_topic[(doc, i)] = (is_atopic, dtopic)

    taken_samples = 0

    it = 0  # iterations
    theta_sampled = {a: 0, d: 0}
    phi_sampled = {a: 0, d: 0}
    pi_sampled = np.zeros((n_docs))

    while taken_samples < samples:
        print("train", it)
        for doc in range(n_docs):  # all documents
            print("train it", it, "doc", doc, "/", n_docs)
            for i, word in enumerate(word_indices(matrix[doc, :])):
                old_is_atopic, old_topic = document_word_topic[(doc, i)]

                if (old_is_atopic):
                    old_author = authors[(doc, i)]
                    cooccurrence_topic_word[a][old_topic, word] -= 1
                    cooccurrence_authordoc_topic[a][(old_author, old_topic)] -= 1
                    occurrence_topic[a][old_topic] -= 1
                    occurrence_author[old_author] -= 1
                    n_words_per_doc[a][doc] -= 1
                else:
                    cooccurrence_topic_word[d][old_topic, word] -= 1
                    cooccurrence_authordoc_topic[d][(doc, old_topic)] -= 1
                    occurrence_topic[d][old_topic] -= 1
                    n_words_per_doc[d][doc] -= 1

                is_atopic, new_author, new_topic = sample_topic(doc, word, n_topics, doc_authors, cooccurrence_topic_word, occurrence_topic, cooccurrence_authordoc_topic, occurrence_author, n_words_per_doc, alpha, beta, delta)

                if (is_atopic):
                    authors[(doc, i)] = new_author
                    cooccurrence_topic_word[a][new_topic, word] += 1
                    cooccurrence_authordoc_topic[a][(new_author, new_topic)] += 1
                    occurrence_topic[a][new_topic] += 1
                    occurrence_author[new_author] += 1
                    n_words_per_doc[a][doc] += 1
                    document_word_topic[(doc, i)] = (is_atopic, new_topic)
                else:
                    cooccurrence_topic_word[d][new_topic, word] += 1
                    cooccurrence_authordoc_topic[d][(doc, new_topic)] += 1
                    occurrence_topic[d][new_topic] += 1
                    n_words_per_doc[d][doc] += 1
                    document_word_topic[(doc, i)] = (is_atopic, new_topic)

        if it >= burn_in:
            it_after_burn_in = it - burn_in
            if (it_after_burn_in % spacing) == 0:
                phi_sampled[a] += topic_phi(a, cooccurrence_topic_word, beta)
                phi_sampled[d] += topic_phi(d, cooccurrence_topic_word, beta)
                theta_sampled[a] += topic_theta(a, cooccurrence_authordoc_topic, alpha)
                theta_sampled[d] += topic_theta(d, cooccurrence_authordoc_topic, alpha)
                pi_sampled += pi(delta, n_words_per_doc)
                taken_samples += 1

        it += 1


    theta_sampled[a] /= taken_samples
    theta_sampled[d] /= taken_samples
    phi_sampled[a] /= taken_samples
    phi_sampled[d] /= taken_samples
    pi_sampled /= taken_samples

    return(theta_sampled, phi_sampled, pi_sampled, chi_sampled)


def classify(test_matrix, test_burn_in, test_samples, test_spacing, n_topics, alpha, beta, delta, eta, phi_sampled):
    n_docs, n_words = test_matrix.shape

    n_words_per_doc = {a : np.zeros((n_docs)), d : np.zeros((n_docs))}
    occurrence_author = np.zeros((n_docs))
    occurrence_topic = {a : np.zeros((n_topics[a])), d : np.zeros((n_topics[d]))}
    cooccurrence_authordoc_topic = {a: np.zeros((n_docs, n_topics[a])), d: np.zeros((n_docs, n_topics[d]))}
    document_atopic_dtopic_ratio = np.zeros((n_docs))
    document_word_topic = {}  # index : (doc, word-index) value: (1 for atopic 0 for dtopic, topic)

    for doc in range(n_docs):
        fic_author = doc
        document_atopic_dtopic_ratio[doc] = np.random.beta(delta[a], delta[d])
        # i is a number between 0 and doc_length-1
        # w is a number between 0 and vocab_size-1
        for i, word in enumerate(word_indices(test_matrix[doc, :])):
            # choose an arbitrary topic as first topic for word i
            is_atopic = np.random.binomial(1, document_atopic_dtopic_ratio[doc])
            if (is_atopic):
                atopic = np.random.randint(n_topics[a])
                occurrence_topic[a][atopic] += 1
                occurrence_author[fic_author] += 1
                document_word_topic[(doc, i)] = (is_atopic, atopic)
                cooccurrence_authordoc_topic[a][(fic_author, atopic)] += 1
                n_words_per_doc[a][doc] += 1
            else:
                dtopic = np.random.randint(n_topics[d])
                cooccurrence_authordoc_topic[d][(doc, dtopic)] += 1
                occurrence_topic[d][dtopic] += 1
                n_words_per_doc[d][doc] += 1
                document_word_topic[(doc, i)] = (is_atopic, dtopic)

    taken_samples = 0

    it = 0  # iterations
    theta_sampled = {a: 0, d: 0}
    pi_sampled = np.zeros((n_docs))

    while taken_samples < test_samples:
        print("classify", it)
        for doc in range(n_docs):  # all documents
            print("classify it", it, "doc", doc, "/", n_docs)
            fic_author = doc
            for i, word in enumerate(word_indices(test_matrix[doc, :])):
                old_is_atopic, old_topic = document_word_topic[(doc, i)]

                if (old_is_atopic):
                    cooccurrence_authordoc_topic[a][(fic_author, old_topic)] -= 1
                    occurrence_topic[a][old_topic] -= 1
                    n_words_per_doc[a][doc] -= 1
                else:
                    cooccurrence_authordoc_topic[d][(doc, old_topic)] -= 1
                    occurrence_topic[d][old_topic] -= 1
                    n_words_per_doc[d][doc] -= 1

                document_dtopics = (cooccurrence_authordoc_topic[d][doc, :] + alpha[d]) / \
                                  (n_words_per_doc[d][doc] + alpha[d] * n_topics[d])

                distribution_d = (delta[d] + n_words_per_doc[d][doc]) * document_dtopics * phi_sampled[d][:, word]

                # normalize to obtain probabilities
                distribution_d = normalise(distribution_d)

                author_atopics = (cooccurrence_authordoc_topic[a][fic_author, :] + alpha[a]) / \
                                (occurrence_author[fic_author] + alpha[a] * n_topics[a])

                distribution_a = (delta[a] + n_words_per_doc[a][doc]) * author_atopics * phi_sampled[a][:, word]

                # normalize to obtain probabilities
                distribution_a = normalise(distribution_a)

                distribution = np.concatenate((distribution_d, distribution_a))

                distribution = normalise(distribution)

                index = np.random.multinomial(1, distribution).argmax()

                is_dtopic = index < n_topics[d]

                if is_dtopic:
                    new_dtopic = index
                    cooccurrence_authordoc_topic[d][(doc, new_dtopic)] += 1
                    occurrence_topic[d][new_dtopic] += 1
                    n_words_per_doc[d][doc] += 1
                    document_word_topic[(doc, i)] = (0, new_dtopic)
                else:
                    new_atopic = index - n_topics[d]
                    cooccurrence_authordoc_topic[a][(fic_author, new_atopic)] += 1
                    occurrence_topic[a][new_atopic] += 1
                    n_words_per_doc[a][doc] += 1
                    document_word_topic[(doc, i)] = (1, new_atopic)

        if it >= test_burn_in:
            it_after_burn_in = it - test_burn_in
            if (it_after_burn_in % test_spacing) == 0:
                theta_sampled[a] += topic_theta(a, cooccurrence_authordoc_topic, alpha)
                theta_sampled[d] += topic_theta(d, cooccurrence_authordoc_topic, alpha)
                pi_sampled += pi(delta, n_words_per_doc)
                taken_samples += 1


        it += 1

    theta_sampled[a] /= taken_samples
    theta_sampled[d] /= taken_samples
    pi_sampled /= taken_samples

    return(theta_sampled, pi_sampled)

def dadt_p(matrix, n_authors, theta, phi, pi_test, chi):
    n_docs, vocab_size = matrix.shape
    candidate_probabilities = np.zeros((n_docs, n_authors))
    for doc in range(n_docs):
        print("deciding doc", doc, "/", n_docs)
        for candidate in range(n_authors):
            text_prob = 0
            for i, word in enumerate(word_indices(matrix[doc, :])):
                theta_vector = {}
                phi_vector = {}

                theta_vector[a] = theta[a][candidate, :]
                phi_vector[a] = phi[a][:, word]
                author_product =  np.dot(theta_vector[a], phi_vector[a])

                theta_vector[d] = theta[d][doc, :]
                phi_vector[d] = phi[d][:, word]
                document_product =  np.dot(theta_vector[d], phi_vector[d])

                text_prob += np.log(pi_test[doc] * author_product + (1-pi_test[doc])*document_product)

            candidate_probabilities[doc, candidate] = np.log(chi[candidate]) + text_prob

    return candidate_probabilities
