#!/usr/bin/python3
import sys, os, re
import numpy as np
from models import *
import math
from multiprocessing.pool import ThreadPool
from itertools import repeat

# DIRECTORY = '../data/nichtschiller'
# DIRECTORY = '../data/blogsprocessed'
DIRECTORY = '../data/judgmentprocessed'
cores = 32

stopwords = []
stopfile = open("../data/stopwords.txt")
for line in stopfile:
    line = line.strip()
    stopwords.append(line)
stopfile.close()

def train_test(model):

    docs_content = {}
    doc_authors = []
    author_ids = {}  # id (int) → author name (string)
    doc_ids = {}

    test_docs_content = {}
    test_doc_authors = []
    test_author_ids = {}  # id (int) → author name (string)
    test_doc_ids = {}

    test_docs_content = {}
    test_doc_authors = []
    test_author_ids = {}  # id (int) → author name (string)
    test_doc_ids = {}

    pattern = re.compile(r'[\W_]+')
    author_pattern = re.compile(r'(\w+)-\d+\.txt')

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

    author_probs = model(matrix, test_matrix, n_authors, vocab, stopwords, doc_authors)

    authors = np.argmax(author_probs, axis = 1)

    accuracy=np.equal(test_doc_authors,authors).mean()

    print(accuracy)

def cross_preprocess():
    print("CROSS PREPROCESS")
    docs_content = {}
    doc_authors = [] #doc_id (int) → author_id (int)
    author_ids = {}  # id (int) → author name (string)
    doc_ids = {} # id (int) → document name (string)
    author_docs = {} # author id (int) -> set of doc ids

    pattern = re.compile(r'[\W_]+')
    author_pattern = re.compile(r'(\w+)-\d+\.txt')

    next_author_id = 0
    doc_id = 0
    next_test_doc_id = 0

    for root, dirs, files in os.walk(DIRECTORY):
        for filename in files:
            print(doc_id)
            match = author_pattern.match(filename)
            if not match:
                print('No match', filename)
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
                        print("multiple authors", doc_authors[doc_id], filename)
                    else:
                        author = doc_authors[doc_id][0]
                        if author not in author_docs:
                            author_docs[author] = set()
                        author_docs[author].add(doc_id)
                    doc_id += 1

    print("END CROSS PREPROCESS")

    return (docs_content, doc_authors, author_docs, doc_ids, author_ids)

def cross_validation(n, models, docs_content, doc_authors, author_docs, _):
    print("CROSS VALIDATION", n)
    author_docs_local = copy.deepcopy(author_docs)
    author_splits = {}

    for author in range(len(author_ids)):
        sample_size =  len(author_docs_local[author])//n
        rest = len(author_docs_local[author]) % n
        for i in range(n):
            if i == 0:
                author_splits[author] = []
            actual_size = sample_size
            if i < rest:
                actual_size += 1
            sample = set(random.sample(author_docs_local[author], actual_size))
            author_docs_local[author] -= sample
            author_splits[author].append(sample)

    numbers = list(range(n))
    with ThreadPool(cores) as p:
        real_author_list = p.starmap(fold_map_function, zip(numbers, repeat(models), repeat(author_ids), repeat(author_splits), repeat(docs_content), repeat(doc_authors)))

    test_authors = {}

    for folddict in real_author_list:
        for model, modeldict in folddict.items():
            if not model in test_authors:
                test_authors[model] = {}
            test_authors[model].update(modeldict)

    print("END CROSS VALIDATION")

    return test_authors

def fold_map_function(i, models, author_ids, author_splits, docs_content, doc_authors):
    print("FOLD", i)
    train_docs_content = {}
    test_docs_content = {}
    train_doc_authors = []
    real_ids = {}
    train_docs_num = 0
    test_docs_num = 0

    for author in range(len(author_ids)):
        for sample_number, sample in enumerate(author_splits[author]):
            for doc_id in sample:
                if(sample_number == i):
                    test_docs_content[test_docs_num] = docs_content[doc_id]
                    real_ids[test_docs_num] = doc_id
                    test_docs_num += 1
                else:
                    train_docs_content[train_docs_num] = docs_content[doc_id]
                    train_doc_authors.append(doc_authors[doc_id])
                    train_docs_num += 1

    (matrix, test_matrix, n_authors, vocab) = preprocess(train_docs_content, test_docs_content, author_ids)

    with ThreadPool(cores) as p:
        author_probs_list = p.starmap(model_help_function, zip(models, repeat(matrix), repeat(test_matrix), repeat(n_authors), repeat(train_doc_authors), repeat(vocab), repeat(stopwords)))


    author_probs_dict = {models[i].__name__: content for i, content in enumerate(author_probs_list)}
    real_authors = {}
    for model in models:
        real_authors[model.__name__] = {}
        for doc, authors in enumerate(author_probs_dict[model.__name__]):
            real_doc_id = real_ids[doc]
            real_authors[model.__name__][real_doc_id] = authors

    print("END FOLD", i)
    return real_authors

def model_help_function(model, matrix, test_matrix, n_authors, train_doc_authors, vocab, stopwords):
    return model(matrix, test_matrix, n_authors, train_doc_authors, vocab, stopwords)

def preprocess(docs_content, test_docs_content, author_ids):

    print("PREPROCESS")

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

    print("END PREPROCESS")

    return(matrix, test_matrix, n_authors, vocab)

def run_chains(models, chains, n_fold, docs_content, doc_authors, author_docs):
    doc_probabilities = 0

    with ThreadPool(cores) as p:
        test_authors_list = p.starmap(cross_validation, zip(repeat(n_fold), repeat(models), repeat(docs_content), repeat(doc_authors), repeat(author_docs), range(chains)))

    for test_authors in test_authors_list:
        for model, rest in test_authors.items():
            num_docs = len(doc_authors)
            num_authors = len(list(test_authors[model].values())[0])
            if np.all(doc_probabilities == 0):
                doc_probabilities = {}
            for doc_id, probs in test_authors[model].items():
                if model not in doc_probabilities:
                    doc_probabilities[model] = np.zeros((num_docs, num_authors))
                doc_probabilities[model][doc_id] += probs

    guessed_authors = {}
    for model, rest in doc_probabilities.items():
        guessed_authors[model] =  np.argmax(rest, axis=1)
    accuracies = {}
    for model in models:
        accuracies[model.__name__] = np.equal(doc_authors, guessed_authors[model.__name__]).mean()

    return accuracies


results = open("results.txt", "w")
chains = 4
n_fold = 10
# model_list = [TOKEN_SVM, LDA_SVM, AT_SVM, AT_P, AT_FA_SVM, AT_FA_P1, AT_FA_P2, DADT_SVM, DADT_P]
model_list = [DADT_P]

(docs_content, doc_authors, author_docs, doc_ids, author_ids) = cross_preprocess()

accuracies = run_chains(model_list, chains, n_fold, docs_content, doc_authors, author_docs)

for model, accuracy in accuracies.items():
    results.write(str(model) + ":" + str(accuracy) + "\n")

results.close()
