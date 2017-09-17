#!/usr/bin/python3
import numpy as np
from scipy.special import gammaln


def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1, p).argmax()


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
        self.occurrence_author = np.zeros(self.n_authors)
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
                self.occurrence_author[author] += 1
                self.topics[(doc, i)] = topic
                self.authors[(doc, i)] = author

    def _conditional_distribution(self, authors, word):
        """
        Conditional distribution (matrix).
        """
        vocab_size = self.cooccur_topic_word.shape[1]

        wt = (self.cooccur_topic_word[:, word] + self.beta) / (self.occurrence_topic + self.beta * vocab_size)

        at = (self.cooccur_author_topic[authors, :] + self.alpha) / (self.occurrence_author[authors].repeat(self.n_topics).reshape(len(authors),
                                                                            self.n_topics) + self.alpha * self.n_topics)

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
        # vocab_size = self.cooccur_topic_word.shape[1]
        #
        # print(self.cooccur_author_topic)
        #
        # ll = self.n_authors * gammaln(self.alpha * self.n_topics)
        # ll -= self.n_authors * self.n_topics * gammaln(self.alpha)
        # ll += self.n_topics * gammaln(self.beta * vocab_size)
        # ll -= self.n_topics * vocab_size * gammaln(self.beta)
        #
        # for ai in range(self.n_authors):
        #     ll += gammaln(self.cooccur_author_topic[ai, :]).sum() - gammaln(self.cooccur_author_topic[ai, :].sum())
        # for ti in range(self.n_topics):
        #     ll += gammaln(self.cooccur_topic_word[ti, :]).sum() - gammaln(self.cooccur_topic_word[ti, :].sum())
        #
        # return ll

        # vocab_size = self.cooccur_topic_word.shape[1]
        # n_docs = self.cooccur_doc_topic.shape[0]
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

    def run(self, doc_authors, matrix, burn_in, samples, spacing):
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
            print('Iteration ', it)
            for doc in range(n_docs):  # all documents
                for i, word in enumerate(word_indices(matrix[doc, :])):  # 1 3, 2 3, 3 3, 4 3, 5 4, 6 4, ...

                    old_topic = self.topics[(doc, i)]
                    old_author = self.authors[(doc, i)]

                    self.cooccur_topic_word[old_topic, word] -= 1
                    self.cooccur_author_topic[old_author, old_topic] -= 1
                    self.occurrence_topic[old_topic] -= 1
                    self.occurrence_author[old_author] -= 1
                    self.number_words_per_doc[doc] -= 1

                    #                     p_z = self._conditional_distribution(doc, word)
                    #                     new_topic = sample_index(p_z)

                    distribution = self._conditional_distribution(self.doc_authors[doc], word)
                    idx = np.random.multinomial(1, distribution).argmax()

                    new_author = self.doc_authors[doc][int(idx / self.n_topics)]
                    new_topic = idx % self.n_topics

                    self.cooccur_author_topic[new_author, new_topic] += 1
                    self.number_words_per_doc[doc] += 1
                    self.cooccur_topic_word[new_topic, word] += 1
                    self.occurrence_topic[new_topic] += 1
                    self.occurrence_author[new_author] += 1
                    self.topics[(doc, i)] = new_topic
                    self.authors[(doc, i)] = new_author
            if it >= burn_in:
                it_after_burn_in = it - burn_in
                if (it_after_burn_in % spacing) == 0:
                    print('    Sampling!')
                    theta += self.theta()
                    phi += self.phi()
                    taken_samples += 1
            it += 1
            print('    Log Likelihood: ', self.loglikelihood())

        theta /= taken_samples
        phi /= taken_samples



        return (theta, phi, self.loglikelihood())

