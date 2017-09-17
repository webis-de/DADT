#!/usr/bin/python3
import numpy as np


def word_indices(vec):
	"""
	Turn a document vector of size vocab_size to a sequence
	of word indices. The word indices are between 0 and
	vocab_size-1. The sequence length is equal to the document length.
	"""
	for idx in vec.nonzero()[0]:  # if word 5 appears 3 times, returns 5 5 5
		for i in range(int(vec[idx])):
			yield idx


def dtopic_distribution(cooccurence_dtopic_word, cooccurence_doc_dtopic, alpha, beta):
	phi = cooccurence_dtopic_word + beta
	phi /= np.sum(phi, axis=1)[:, np.newaxis]

	theta = cooccurence_doc_dtopic + alpha
	theta /= np.sum(theta, axis=1)[:, np.newaxis]

	return phi, theta

def atopic_distribution(cooccurence_atopic_word, cooccurence_author_atopic, alpha, beta):
	# word topic distribution
	phi = cooccurence_atopic_word + beta
	phi /= np.sum(phi, axis=1)[:, np.newaxis]

	# author topic distribution
	theta = cooccurence_author_atopic + alpha
	theta /= np.sum(theta, axis=1)[:, np.newaxis]

	return phi, theta

def sample_dtopic(doc, word, num_dtopics, cooccur_dtopic_word, occurrence_dtopic, cooccur_doc_dtopic, number_dwords_per_doc, alpha, beta):
	vocab_size = cooccur_dtopic_word.shape[1]

	word_topics = (cooccur_dtopic_word[:, word] + beta) / \
				  (occurrence_dtopic + beta * vocab_size)

	document_topics = (cooccur_doc_dtopic[doc, :] + alpha) / \
					  (number_dwords_per_doc[doc] + alpha * num_dtopics)

	distribution = document_topics * word_topics
	# normalize to obtain probabilities
	distribution /= np.sum(distribution)

	new_dtopic = np.random.multinomial(1, distribution).argmax()
	return new_dtopic


def sample_atopic_and_author(doc, word, authors, doc_authors, cooccur_atopic_word, occurrence_atopic, cooccur_author_atopic, occurrence_author, num_atopics, alpha, beta):
	vocab_size = cooccur_atopic_word.shape[1]

	word_topics = (cooccur_atopic_word[:, word] + beta) / (occurrence_atopic + beta * vocab_size)

	author_topics = (cooccur_author_atopic[authors, :] + alpha) / (
		occurrence_author[authors].repeat(num_atopics).reshape(len(authors), num_atopics) + alpha * num_atopics)

	distribution = author_topics * word_topics
	# reshape into a looong vector
	distribution = distribution.reshape(len(authors) * num_atopics)
	# normalize to obtain probabilities
	distribution /= distribution.sum()

	idx = np.random.multinomial(1, distribution).argmax()

	new_author = doc_authors[doc][int(idx / num_atopics)]
	new_topic = idx % num_atopics

	return (new_topic, new_author)


def learn(matrix, doc_authors, num_dtopics, num_atopics, num_authors, alpha_a, beta_a, alpha_d, beta_d, eta, delta_A, delta_D, burn_in, samples, spacing):
	num_docs, num_words = matrix.shape

	# Initialize matricies
	coocurrence_dtopic_word = np.zeros((num_dtopics, num_words))
	coocurrence_atopic_word = np.zeros((num_atopics, num_words))
	coocurrence_author_atopic = np.zeros((num_authors, num_atopics))
	prior_author = np.zeros((num_authors))  # only used in classifier
	coocurrence_document_dtopic = np.zeros((num_docs, num_dtopics))

	occurrence_atopic = np.zeros((num_atopics))
	occurrence_dtopic = np.zeros((num_dtopics))
	occurrence_author = np.zeros((num_authors))

	num_dwords_per_doc = np.zeros((num_docs))

	document_atopic_dtopic_ratio = np.zeros((num_docs))
	document_word_topic = {}  # index : (doc, word-index) value: (1 for atopic 0 for dtopic, topic)
	authors = {}

	for doc in range(num_docs):
		document_atopic_dtopic_ratio[doc] = np.random.beta(delta_A, delta_D)
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
				coocurrence_atopic_word[(atopic, word)] += 1
				coocurrence_author_atopic[(author, atopic)] += 1
				authors[(doc, i)] = author
			else:
				dtopic = np.random.randint(num_dtopics)
				coocurrence_dtopic_word[(dtopic, word)] += 1
				coocurrence_document_dtopic[(doc, dtopic)] += 1
				occurrence_dtopic[dtopic] += 1
				num_dwords_per_doc[doc] += 1
				document_word_topic[(doc, i)] = (is_atopic, dtopic)

	taken_samples = 0

	it = 0  # iterations
	while taken_samples < samples:
		print('Iteration ', it)
		for doc in range(num_docs):  # all documents
			for i, word in enumerate(word_indices(matrix[doc, :])):
				old_is_atopic, old_topic = document_word_topic[(doc, i)]

				if (old_is_atopic):
					old_author = authors[(doc, i)]
					coocurrence_atopic_word[(old_author, word)] -= 1
					coocurrence_author_atopic[(old_author, old_topic)] -= 1
					occurrence_atopic[old_topic] -= 1
					occurrence_author[old_author] -= 1
				else:
					coocurrence_dtopic_word[(old_topic, word)] -= 1
					coocurrence_document_dtopic[(doc, old_topic)] -= 1
					occurrence_dtopic[old_topic] -= 1
					num_dwords_per_doc[doc] -= 1

				is_atopic = np.random.binomial(1, document_atopic_dtopic_ratio[doc])

				if (is_atopic):
					new_atopic, new_author = sample_atopic_and_author(doc, word, authors, doc_authors, coocurrence_atopic_word, occurrence_atopic,  coocurrence_author_atopic, occurrence_author, num_atopics, alpha_a, beta_a)

					occurrence_atopic[new_atopic] += 1
					occurrence_author[new_author] += 1
					document_word_topic[(doc, i)] = (is_atopic, new_atopic)
					coocurrence_atopic_word[(new_atopic, word)] += 1
					coocurrence_author_atopic[(new_author, new_atopic)] += 1
					authors[(doc, i)] = new_author
				else:
					new_dtopic = sample_dtopic(doc, word, num_dtopics, coocurrence_dtopic_word, occurrence_dtopic, coocurrence_document_dtopic, num_dwords_per_doc, alpha_d, beta_d)

					coocurrence_dtopic_word[(new_dtopic, word)] += 1
					coocurrence_document_dtopic[(doc, new_dtopic)] += 1
					occurrence_dtopic[new_dtopic] += 1
					num_dwords_per_doc[doc] += 1
					document_word_topic[(doc, i)] = (is_atopic, new_dtopic)

		if it >= burn_in:
			it_after_burn_in = it - burn_in
			if (it_after_burn_in % spacing) == 0:
				print('    Sampling!')
				# TODO sample
				taken_samples += 1
		it += 1


def classify(matrix, burn_in, samples, spacing, alpha, beta, eta, delta_A, delta_D):
	print("error: not yet implemented")
	return [0]
