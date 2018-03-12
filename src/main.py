#!/usr/bin/python3
import sys, os, re
import numpy as np
from models import *
import math
from multiprocessing.pool import ThreadPool
from itertools import repeat
import yappi

# DIRECTORY = '../data/nichtschiller/'
# DIRECTORY = '../data/blogsprocessed/'
# DIRECTORY = '../data/judgmentprocessed/'
# DIRECTORY = '../data/c10processed/'
# DIRECTORY = '../data/pan11processed/'
DIRECTORY = '../data/pan12processed-tiny/'
cores = 32

stopwords = []
stopfile = open("../data/stopwords.txt")
for line in stopfile:
    line = line.strip()
    stopwords.append(line)
stopfile.close()

def read_files(folder, author_ids):
    pattern = re.compile(r'[\W_]+')
    author_pattern = re.compile(r'(\w+)-\d+\.txt')
    files = os.listdir(folder)
    docs_content = []
    doc_authors = []
    for filename in files:
        match = author_pattern.match(filename)
        if not match:
            print('No match in', filename)
            continue

        author = match.group(1)

        if author not in author_ids:
            author_id = len(author_ids)
            author_ids[author] = author_id
        else:
            author_id = author_ids[author]

        with open(folder + filename) as f:
            content = []
            for line in f:
                words = [pattern.sub('', w.lower()) for w in line.split()]
                content.extend(words)
            docs_content.append(content)
            doc_authors.append([author_id])

    return docs_content, doc_authors, author_ids

def cross_validation(n, models, docs_content, doc_authors, _):
    print("CROSS VALIDATION", n)
    author_docs = [set([doc for doc, author in enumerate(doc_authors) if author == a]) for a in sorted(set(doc_authors))]
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

    return test_authors, STAT_OBJS

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

    (matrix, test_matrix, n_authors, vocab) = build_data(train_docs_content, test_docs_content, author_ids)

    author_probs_dict = model_map(models, matrix, test_matrix, n_authors, train_doc_authors, vocab, stopwords)

    real_authors = {}
    for model in models:
        real_authors[model.__name__] = {}
        for doc, authors in enumerate(author_probs_dict[model.__name__]):
            real_doc_id = real_ids[doc]
            real_authors[model.__name__][real_doc_id] = authors

    print("END FOLD", i)
    return real_authors

def model_map(models, matrix, test_matrix, n_authors, train_doc_authors, vocab, stopwords, _):
    with ThreadPool(cores) as p:
        author_probs_list = p.starmap(model_help_function, zip(models, repeat(matrix), repeat(test_matrix), repeat(n_authors), repeat(train_doc_authors), repeat(vocab), repeat(stopwords)))
    author_probs_dict = {models[i].__name__: content for i, content in enumerate(author_probs_list)}
    return author_probs_dict

def model_help_function(model, matrix, test_matrix, n_authors, train_doc_authors, vocab, stopwords):
    result = model(matrix, test_matrix, n_authors, train_doc_authors, vocab, stopwords)

    return result

def build_data(train_docs_content, test_docs_content, author_ids):

    print("PREPROCESS")

    vocab = set()
    for doc, content in enumerate(train_docs_content):
        for word in content:
            if len(word) > 1:
                vocab.add(word)

    for doc, content in enumerate(test_docs_content):
        for word in content:
            if len(word) > 1:
                vocab.add(word)

    vocab = list(vocab)
    lookupvocab = dict([(word, index) for (index, word) in enumerate(vocab)])

    matrix = np.zeros((len(train_docs_content), len(vocab)))

    test_matrix = np.zeros((len(test_docs_content), len(vocab)))

    for doc, content in enumerate(train_docs_content):
        for word in content:
            if len(word) > 1:
                matrix[doc, lookupvocab[word]] += 1

    for doc, content in enumerate(test_docs_content):
        for word in content:
            if len(word) > 1:
                test_matrix[doc, lookupvocab[word]] += 1

    n_authors = len(author_ids)

    print("END PREPROCESS")

    return(matrix, test_matrix, n_authors, vocab)

def run_chains(function, arguments, test_doc_authors):

    print('run chains')
    doc_probabilities = 0

    with ThreadPool(cores) as p:
        test_authors_list = p.starmap(function, arguments)

    for test_authors in test_authors_list:
        for model, rest in test_authors.items():
            num_docs = len(test_doc_authors)
            num_authors = len(rest[0])
            if np.all(doc_probabilities == 0):
                doc_probabilities = {}
            for doc_id, probs in enumerate(rest):
                if model not in doc_probabilities:
                    doc_probabilities[model] = np.zeros((num_docs, num_authors))
                doc_probabilities[model][doc_id] += probs

    guessed_authors = {}
    for model, rest in doc_probabilities.items():
        guessed_authors[model] =  np.argmax(rest, axis=1)

    accuracies = {}
    for model, model_authors in guessed_authors.items():
        print(test_doc_authors, model_authors)
        accuracies[model] = np.equal(np.transpose(test_doc_authors), model_authors).mean()
    return accuracies

def cross_main(chains, n_fold, models):
    author_ids = {}
    docs_content, doc_authors, author_ids = read_files(DIRECTORY, author_ids)
    accuracies = run_chains(cross_validation, zip(repeat(models), repeat(docs_content), repeat(doc_authors), range(chains)), doc_authors)
    return(accuracies)

def train_main(chains, n_fold, models):
    print("train main")
    author_ids = {} # schiller -> 1

    train_docs_content, train_doc_authors, author_ids = read_files(DIRECTORY + 'train/', author_ids)
    test_docs_content, test_doc_authors, author_ids = read_files(DIRECTORY + 'test/', author_ids)

    (matrix, test_matrix, n_authors, vocab) = build_data(train_docs_content, test_docs_content, author_ids)
    accuracies = run_chains(model_map, zip(repeat(models), repeat(matrix), repeat(test_matrix), repeat(n_authors), repeat(train_doc_authors), repeat(vocab), repeat(stopwords), range(chains)), test_doc_authors)

    return(accuracies)

chains = 4
n_fold = 10
models = [DADT_P]
# models = [TOKEN_SVM, LDA_SVM, AT_SVM, AT_P, AT_FA_SVM, AT_FA_P1, AT_FA_P2, DADT_SVM, DADT_P]


# accuracies = cross_main(chains, n_fold, models)
yappi.start()

accuracies = train_main(chains, n_fold, models)

yappi.stop()
yappi.get_func_stats().print_all()


dir_re = re.compile(r'.*\/data\/(.*?)\/')
match = dir_re.match(DIRECTORY)
dir_name = match.group(1)
results = open("results_" + dir_name + "_" + "-".join([x.__name__ for x in models]) + ".txt", "w")
for model, accuracy in accuracies.items():
    results.write(str(model) + " : " + str(accuracy) + "\n")
results.close()
