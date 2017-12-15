#!/usr/bin/python3
import numpy as np
from math import lgamma

gammaln = np.vectorize(lgamma)

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:  # if word 5 appears 3 times, returns 5 5 5
        for i in range(int(vec[idx])):
            yield idx


def log_multi_beta(alpha):
    """
    Logarithm of the multinomial beta function.
    """
    return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))


class AtSampler(object):
    def __init__(self, n_topics, n_authors, alpha, beta):
        """
        n_topics: desired number of topics
        alpha: a vector of length n_topics
        beta: a vector of length vocab_size
        """
        self.n_topics = n_topics
        self.n_authors = n_authors
        self.alpha = alpha
        self.beta = beta

    def _initialize(self, doc_authors, matrix):
        n_docs, vocab_size = matrix.shape

        self.doc_authors = doc_authors
        self.cooccur_author_topic = np.zeros((self.n_authors, self.n_topics))
        self.cooccur_topic_word = np.zeros((self.n_topics, vocab_size))
        self.number_words_per_doc = np.zeros(n_docs)
        self.occurrence_topic = np.zeros(self.n_topics)
        self.num_words_per_author = np.zeros(self.n_authors)
        self.topics = {}
        self.authors = {}

        for doc in range(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, word in enumerate(word_indices(matrix[doc, :])):
                # choose an arbitrary topic as first topic for word i
                topic = np.random.randint(self.n_topics)
                author = doc_authors[doc][np.random.randint(len(doc_authors[doc]))]
                self.cooccur_author_topic[author, topic] += 1
                self.number_words_per_doc[doc] += 1
                self.cooccur_topic_word[topic, word] += 1
                self.occurrence_topic[topic] += 1
                self.num_words_per_author[author] += 1
                self.topics[(doc, i)] = topic
                self.authors[(doc, i)] = author

    def _conditional_distribution(self, authors, word):
        """
        Conditional distribution (matrix).
        """
        vocab_size = self.cooccur_topic_word.shape[1]


        # at_complex = (self.cooccur_author_topic[authors, :] + self.alpha) / (self.num_words_per_author[authors].repeat(self.n_topics).reshape(len(authors),
        #                                                                     self.n_topics) + self.alpha * self.n_topics)
        at = (self.cooccur_author_topic[authors, :] + self.alpha)
        at /= np.sum(at, axis=1)[:, np.newaxis]

        # wt_complex = (self.cooccur_topic_word[:, word] + self.beta) / (self.occurrence_topic + self.beta * vocab_size)

        wt = (self.cooccur_topic_word + self.beta)
        wt /= np.sum(wt, axis=1)[:, np.newaxis]
        wt = wt[:, word]


        pdf = at * wt
        # reshape into a looong vector
        pdf = pdf.reshape(len(authors) * self.n_topics)
        # normalize to obtain probabilities
        pdf /= pdf.sum()
        return pdf

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """

        lik = 0

        for topic in range(self.n_topics):
            lik += log_multi_beta(self.cooccur_topic_word[topic, :] + self.beta)
            lik -= log_multi_beta(self.beta)

        for author in range(self.n_authors):
            lik += log_multi_beta(self.cooccur_author_topic[author, :] + self.alpha)
            lik -= log_multi_beta(self.alpha)

        return lik


    def phi(self):
	    # word topic distribution
        num = self.cooccur_topic_word + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def theta(self):
        # author topic distribution
        num = self.cooccur_author_topic + self.alpha
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def train(self, doc_authors, matrix, burn_in, samples, spacing):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape
        self._initialize(doc_authors, matrix)
        theta = 0
        phi = 0
        taken_samples = 0

        it = 0  # iterations
        while taken_samples < samples:
            for doc in range(n_docs):  # all documents
                for i, word in enumerate(word_indices(matrix[doc, :])):  # 1 3, 2 3, 3 3, 4 3, 5 4, 6 4, ...

                    old_topic = self.topics[(doc, i)]
                    old_author = self.authors[(doc, i)]

                    self.cooccur_topic_word[old_topic, word] -= 1
                    self.cooccur_author_topic[old_author, old_topic] -= 1
                    self.occurrence_topic[old_topic] -= 1
                    self.num_words_per_author[old_author] -= 1
                    self.number_words_per_doc[doc] -= 1

                    distribution = self._conditional_distribution(self.doc_authors[doc], word)
                    idx = np.random.multinomial(1, distribution).argmax()

                    new_author = self.doc_authors[doc][int(idx / self.n_topics)]
                    new_topic = idx % self.n_topics

                    self.cooccur_author_topic[new_author, new_topic] += 1
                    self.number_words_per_doc[doc] += 1
                    self.cooccur_topic_word[new_topic, word] += 1
                    self.occurrence_topic[new_topic] += 1
                    self.num_words_per_author[new_author] += 1
                    self.topics[(doc, i)] = new_topic
                    self.authors[(doc, i)] = new_author
            if it >= burn_in:
                it_after_burn_in = it - burn_in
                if (it_after_burn_in % spacing) == 0:
                    theta += self.theta()
                    phi += self.phi()
                    taken_samples += 1
            it += 1

        theta /= taken_samples
        phi /= taken_samples

        return (theta, phi)

    def classify(self, matrix, phi, burn_in, samples, spacing):
        n_docs, vocab_size = matrix.shape
        thetas = 0
        n_docs, vocab_size = matrix.shape

        self.cooccur_author_topic = np.zeros((self.n_authors, self.n_topics))
        self.number_words_per_doc = np.zeros(n_docs)
        self.occurrence_topic = np.zeros(self.n_topics)
        self.num_words_per_author = np.zeros(self.n_authors)
        self.topics = {}

        for doc in range(n_docs):
            author = doc
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, word in enumerate(word_indices(matrix[doc, :])):
                # choose an arbitrary topic as first topic for word i
                topic = np.random.randint(self.n_topics)
                self.cooccur_author_topic[author, topic] += 1
                self.number_words_per_doc[doc] += 1
                self.occurrence_topic[topic] += 1
                self.num_words_per_author[author] += 1
                self.topics[(doc, i)] = topic
        theta = 0
        taken_samples = 0

        it = 0  # iterations
        while taken_samples < samples:
            for doc in range(n_docs):  # all documents
                author = doc
                for i, word in enumerate(word_indices(matrix[doc, :])):  # 1 3, 2 3, 3 3, 4 3, 5 4, 6 4, ...

                    old_topic = self.topics[(doc, i)]

                    self.cooccur_author_topic[author, old_topic] -= 1
                    self.occurrence_topic[old_topic] -= 1
                    self.number_words_per_doc[doc] -= 1

                    distribution = ((self.cooccur_author_topic[0, :] + self.alpha) / \
                                    (self.num_words_per_author[0] + self.alpha * self.n_topics) * phi[:, word])
                    distribution /= np.sum(distribution)
                    new_topic = np.random.multinomial(1, distribution).argmax()

                    self.cooccur_author_topic[author, new_topic] += 1
                    self.number_words_per_doc[doc] += 1
                    self.occurrence_topic[new_topic] += 1
                    self.topics[(doc, i)] = new_topic

            if it >= burn_in:
                it_after_burn_in = it - burn_in
                if (it_after_burn_in % spacing) == 0:
                    theta += self.theta()
                    taken_samples += 1
            it += 1

        theta /= taken_samples

        return (theta)

    def at_p(self, phi, theta, matrix):
        n_docs, vocab_size = matrix.shape
        candidate_probabilities = np.zeros((n_docs, self.n_authors))

        for doc in range(n_docs):  # all documents
            for candidate in range(self.n_authors):
                text_prob = 0
                for i, word in enumerate(word_indices(matrix[doc, :])):
                    theta_vector = theta[candidate, :]
                    phi_vector = phi[:, word]
                    product =  np.dot(theta_vector, phi_vector)
                    text_prob += np.log(product)

                candidate_probabilities[doc, candidate] = text_prob

        return candidate_probabilities

    def at_fa_p2(self, phi, theta_r, matrix, samples, burn_in, spacing):
        n_docs, vocab_size = matrix.shape
        candidate_probabilities = np.zeros((n_docs, self.n_authors))

        CANDIDATE_AUTHOR = 0
        FICTITIOUS_AUTHOR = 1

        for candidate in range(self.n_authors):
            cooccur_fic_author_topic = np.zeros(self.n_topics)
            number_words_per_doc = np.zeros(n_docs)
            occurrence_topic = np.zeros(self.n_topics)
            num_words_of_fic_author = 0
            topics = {}
            authors = {}

            for doc in range(n_docs):
                # i is a number between 0 and doc_length-1
                # w is a number between 0 and vocab_size-1
                for i, word in enumerate(word_indices(matrix[doc, :])):
                    # choose an arbitrary topic as first topic for word i
                    topic = np.random.randint(self.n_topics)
                    author = CANDIDATE_AUTHOR if np.random.randint(2) else FICTITIOUS_AUTHOR

                    topics[(doc, i)] = topic
                    authors[(doc, i)] = author

                    if FICTITIOUS_AUTHOR:
                        cooccur_fic_author_topic[topic] += 1
                        occurrence_topic[topic] += 1
                        number_words_per_doc[doc] += 1
                        num_words_of_fic_author += 1

            theta_fic = 0
            taken_samples = 0

            it = 0  # iterations
            while taken_samples < samples:
                for doc in range(n_docs):  # all documents
                    for i, word in enumerate(word_indices(matrix[doc, :])):  # 1 3, 2 3, 3 3, 4 3, 5 4, 6 4, ...

                        old_topic = topics[(doc, i)]
                        old_author = authors[(doc, i)]

                        if old_author == FICTITIOUS_AUTHOR:
                            cooccur_fic_author_topic[old_topic] -= 1
                            occurrence_topic[old_topic] -= 1
                            number_words_per_doc[doc] -= 1
                            num_words_of_fic_author -= 1

                        # vocab_size = self.cooccur_topic_word.shape[1]

                        theta_fake = cooccur_fic_author_topic + self.alpha
                        theta_fake /= theta_fake.sum()

                        distribution_real = np.multiply(theta_r[candidate], phi[:, word])
                        distribution_fake = np.multiply(theta_fake, phi[:, word])

                        distribution = np.concatenate((distribution_real, distribution_fake))
                        # normalize to obtain probabilities
                        distribution /= distribution.sum()

                        idx = np.random.multinomial(1, distribution).argmax()

                        if idx >= self.n_topics:
                            new_topic = idx - self.n_topics
                            new_author = FICTITIOUS_AUTHOR

                            cooccur_fic_author_topic[new_topic] += 1
                            number_words_per_doc[doc] += 1
                            occurrence_topic[new_topic] += 1
                            num_words_of_fic_author += 1
                        else:
                            new_topic = idx
                            new_author = CANDIDATE_AUTHOR

                        topics[(doc, i)] = new_topic
                        authors[(doc, i)] = new_author

                if it >= burn_in:
                    it_after_burn_in = it - burn_in
                    if (it_after_burn_in % spacing) == 0:
                        theta_fic += theta_fake
                        taken_samples += 1
                it += 1

            theta_fic = theta_fake / taken_samples

            for doc in range(n_docs):  # all documents
                text_prob = 0
                for i, word in enumerate(word_indices(matrix[doc, :])):
                    theta_vector = theta_r[candidate, :]
                    phi_vector = phi[:, word]
                    product_real =  np.dot(theta_vector, phi_vector)
                    product_fic = np.dot(theta_fic, phi_vector)
                    sum_prob = product_real + product_fic

                    text_prob += np.log(sum_prob)

                candidate_probabilities[doc, candidate] = text_prob

        return np.exp(candidate_probabilities)
