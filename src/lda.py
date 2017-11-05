import numpy as np
from math import lgamma

gammaln = np.vectorize(lgamma)

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


class LDA(object):
    def __init__(self, n_topics, alpha, beta):
        """
        n_topics: desired number of topics
        alpha: a vector of length n_topics
        beta: a vector of length vocab_size
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta

    def _initialize(self, matrix):
        n_docs, vocab_size = matrix.shape

        self.cooccur_doc_topic = np.zeros((n_docs, self.n_topics))
        self.cooccur_topic_word = np.zeros((self.n_topics, vocab_size))
        self.number_words_per_doc = np.zeros(n_docs)
        self.occurrence_topic = np.zeros(self.n_topics)
        self.topics = {}

        for doc in range(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, word in enumerate(word_indices(matrix[doc, :])):
                # choose an arbitrary topic as first topic for word i
                topic = np.random.randint(self.n_topics)
                self.cooccur_doc_topic[doc, topic] += 1
                self.number_words_per_doc[doc] += 1
                self.cooccur_topic_word[topic, word] += 1
                self.occurrence_topic[topic] += 1
                self.topics[(doc, i)] = topic

    def _conditional_distribution(self, doc, word):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.cooccur_topic_word.shape[1]

        left = (self.cooccur_topic_word[:, word] + self.beta) / \
               (self.occurrence_topic + self.beta * vocab_size)

#         wt = (self.cooccur_topic_word + self.beta)
#         wt /= np.sum(wt, axis=1)[:, np.newaxis]
#         wt = wt[:, word]

        right = (self.cooccur_doc_topic[doc, :] + self.alpha) / \
                (self.number_words_per_doc[doc] + self.alpha * self.n_topics)

#         dt = (self.cooccur_doc_topic[doc, :] + self.alpha)
#         dt /= np.sum(dt, axis=1)[:, np.newaxis]

#         p_z = wt * dt
        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.cooccur_topic_word.shape[1]
        n_docs = self.cooccur_doc_topic.shape[0]
        lik = 0

        for topic in range(self.n_topics):
            lik += log_multi_beta(self.cooccur_topic_word[topic, :] + self.beta)
            lik -= log_multi_beta(self.beta)

        for doc in range(n_docs):
            lik += log_multi_beta(self.cooccur_doc_topic[doc, :] + self.alpha)
            lik -= log_multi_beta(self.alpha)

        return lik

    def phi(self):
        # word topic distribution
        num = self.cooccur_topic_word + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def theta(self):
        # doc topic distribution
        num = self.cooccur_doc_topic + self.alpha
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def train(self, matrix, burn_in, samples, spacing):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape

        self._initialize(matrix)
        theta = 0
        phi = 0
        taken_samples = 0

        it = 0  # iterations
        while taken_samples < samples:
            print('Iteration ', it)
            for doc in range(n_docs):  # all documents
                for i, word in enumerate(word_indices(matrix[doc, :])):  # 1 3, 2 3, 3 3, 4 3, 5 4, 6 4, ...
                    old_topic = self.topics[(doc, i)]
                    self.cooccur_doc_topic[doc, old_topic] -= 1
                    self.number_words_per_doc[doc] -= 1
                    self.cooccur_topic_word[old_topic, word] -= 1
                    self.occurrence_topic[old_topic] -= 1

                    p_z = self._conditional_distribution(doc, word)
                    new_topic = sample_index(p_z)

                    self.cooccur_doc_topic[doc, new_topic] += 1
                    self.number_words_per_doc[doc] += 1
                    self.cooccur_topic_word[new_topic, word] += 1
                    self.occurrence_topic[new_topic] += 1
                    self.topics[(doc, i)] = new_topic
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

    def classify(self, matrix, phi, burn_in, samples, spacing, chains):
        n_docs, vocab_size = matrix.shape
        thetas = 0
        for c in range(chains):
            self._initialize(matrix)
            theta = 0
            taken_samples = 0

            it = 0  # iterations
            while taken_samples < samples:
                print('Iteration ', it)
                for doc in range(n_docs):  # all documents
                    for i, word in enumerate(word_indices(matrix[doc, :])):  # 1 3, 2 3, 3 3, 4 3, 5 4, 6 4, ...
                        old_topic = self.topics[(doc, i)]
                        self.cooccur_doc_topic[doc, old_topic] -= 1
                        self.number_words_per_doc[doc] -= 1
                        self.occurrence_topic[old_topic] -= 1

                        distribution = ((self.cooccur_doc_topic[doc, :] + self.alpha) / \
                                        (self.number_words_per_doc[doc] + self.alpha * self.n_topics) * phi[:, word])
                        distribution /= np.sum(distribution)
                        new_topic = sample_index(distribution)

                        self.cooccur_doc_topic[doc, new_topic] += 1
                        self.number_words_per_doc[doc] += 1
                        self.occurrence_topic[new_topic] += 1
                        self.topics[(doc, i)] = new_topic
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
            thetas += theta

        theta /= chains
        return (theta, self.loglikelihood())
