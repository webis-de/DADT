#!/usr/bin/python3
import sys, os, re
import numpy as np
from models import *
import math

DIRECTORY = '../data/nichtschiller'


def train_test():

    print('Reading corpus')

    docs_content = {}
    doc_authors = []
    author_ids = {}  # id (int) → author name (string)
    doc_ids = {}

    test_docs_content = {}
    test_doc_authors = []
    test_author_ids = {}  # id (int) → author name (string)
    test_doc_ids = {}

    stopwords = []
    stopfile = open("../data/germanST.txt")
    for line in stopfile:
        line = line.strip()
        stopwords.append(line)
    stopfile.close()

    test_docs_content = {}
    test_doc_authors = []
    test_author_ids = {}  # id (int) → author name (string)
    test_doc_ids = {}

    pattern = re.compile(r'[\W_]+')
    author_pattern = re.compile(r'([a-zäöüA-ZÄÖÜ\-]+)\d+\.txt')

    next_author_id = 0
    doc_id = 0
    next_test_doc_id = 0

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
                    test_doc_ids[next_test_doc_id] = filename
                    test_docs_content[next_test_doc_id] = content
                    test_doc_authors.append(author_id)
                    next_test_doc_id += 1

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
                    doc_ids[doc_id] = filename
                    docs_content[doc_id] = content
                    doc_authors.append(doc_author_ids)
                    doc_id += 1

    (matrix, test_matrix, n_authors, vocab) = preprocess(docs_content, test_docs_content, author_ids)

    author_guess = run_model(matrix, test_matrix, n_authors, vocab, stopwords, doc_authors)

    for text in range(len(test_docs_content.items())):
        print(test_doc_ids[text], author_ids[author_guess[text]])

    accuracy=np.equal(test_doc_authors,author_guess).mean()

    print(accuracy)

def cross_validation(n):

    print('Reading corpus')

    docs_content = {}
    doc_authors = [] #doc_id (int) → author_id (int)
    author_ids = {}  # id (int) → author name (string)
    doc_ids = {} # id (int) → document name (string)
    author_docs = {} # author id (int) -> set of doc ids

    stopwords = []
    stopfile = open("../data/germanST.txt")
    for line in stopfile:
        line = line.strip()
        stopwords.append(line)
    stopfile.close()

    pattern = re.compile(r'[\W_]+')
    author_pattern = re.compile(r'([a-zäöüA-ZÄÖÜ\-]+)\d+\.txt')

    next_author_id = 0
    doc_id = 0
    next_test_doc_id = 0

    for root, dirs, files in os.walk(DIRECTORY):
        for filename in files:
            match = author_pattern.match(filename)
            if not match:
                print('No match')
                continue
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
                    doc_ids[doc_id] = filename
                    docs_content[doc_id] = content
                    doc_authors.append(doc_author_ids)
                    if len(doc_authors[doc_id]) > 1:
                        print("multiple authors")
                    else:
                        author = doc_authors[doc_id][0]
                        if author not in author_docs:
                            author_docs[author] = set()
                        author_docs[author].add(doc_id)
                    doc_id += 1

    author_splits = {}

    for author in range(len(author_ids)):
        sample_size =  len(author_docs[author])//n
        rest = len(author_docs[author]) % n
        for i in range(n):
            if i == 0:
                author_splits[author] = []
            actual_size = sample_size
            if i < rest:
                actual_size += 1
            sample = set(random.sample(author_docs[author], actual_size))
            author_docs[author] -= sample
            author_splits[author].append(sample)

    accuracies = []

    for i in range(n):
        train_docs_content = {}
        test_docs_content = {}
        test_doc_authors = []
        train_doc_authors = []
        train_docs_num = 0
        test_docs_num = 0
        for author in range(len(author_ids)):
            for sample_number, sample in enumerate(author_splits[author]):
                for doc_id in sample:
                    if(sample_number == i):
                        test_docs_content[test_docs_num] = docs_content[doc_id]
                        test_doc_authors.append(doc_authors[doc_id])
                        test_docs_num += 1
                    else:
                        train_docs_content[train_docs_num] = docs_content[doc_id]
                        train_doc_authors.append(doc_authors[doc_id])
                        train_docs_num += 1

        (matrix, test_matrix, n_authors, vocab) = preprocess(train_docs_content, test_docs_content, author_ids)

        print(matrix.shape)

        author_guess = run_model(matrix, test_matrix, n_authors, vocab, stopwords, train_doc_authors)

        accuracy=np.equal(test_doc_authors,author_guess).mean()
        print("Accuracy iteration", i, accuracy)
        accuracies.append(accuracy)

    print("Overall accuracy", np.mean(accuracies))

def preprocess(docs_content, test_docs_content, author_ids):

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

    return(matrix, test_matrix, n_authors, vocab)

def run_model(matrix, test_matrix, n_authors, vocab, stopwords, doc_authors):
    # author_guess = TOKEN_SVM(matrix, test_matrix, n_authors, doc_authors)
    author_guess = LDA_SVM(matrix, test_matrix, n_authors, doc_authors)
    # author_guess = AT_SVM(matrix, test_matrix, n_authors, doc_authors)
    # author_guess = AT_P(matrix, test_matrix, n_authors, doc_authors)
    # author_guess = AT_FA_SVM(matrix, test_matrix, n_authors, doc_authors)
    # author_guess = AT_FA_P1(matrix, test_matrix, n_authors, doc_authors)
    # author_guess = AT_FA_P2(matrix, test_matrix, n_authors, doc_authors)
    # author_guess = DADT_SVM(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords)
    # author_guess = DADT_P(matrix, test_matrix, n_authors, doc_authors, vocab, stopwords)

    return author_guess

# train_test()
cross_validation(2)
