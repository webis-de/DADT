"""
GIBBS SAMPLING IMPLEMENTATION FOR LATENT DIRICHLET ALLOCATION (2003)
IMPLEMENTED BY CHANG-UK, PARK
DATA FORMAT: "DocID\t WordID\t FREQUENCY\n"
"""

import random
import sys

from scipy.special import gammaln, psi

import numpy as np

class Sampler(object):
    def __init__(self, pathData, ntopics, alpha, beta, header = False):
        self.header = header
        self.TOPICS = ntopics                                       # NUMBER OF TOPICS
        self.documents = {}                                         # TRAINING DATA: {DocID: [WordID1, WordID1, WordID2, ...]}
        self.doc_to_index = {}                                              # MAP DOCUMENT INTO INDEX: self.doc_to_index = {DocID: INDEX}
        self.word_to_index = {}                                              # MAP WORD INTO INDEX: self.word_to_index = {VocabID: INDEX}
        self.DOCS = 0                                               # NUMBER OF DOCUMENTS
        self.VOCABS = 0                                             # NUMBER OF VOCABULARY WORDS
        self.alpha = alpha                                # np.random.gamma(0.1, 1)
        self.beta = beta                                         # np.random.gamma(0.1, 1)
        data = open(pathData, "r")
        [self.LoadData(r) for r in data.read().split("\n")]         # LOAD TRAINING DATA INTO 'self.documents'
        for doc in self.documents:
            random.shuffle(self.documents[doc])                     # SHUFFLE WORDS IN EACH DOCUMENT
        data.close()
        self.theta = np.zeros((self.DOCS, self.TOPICS))             # SPACE FOR THETA MATRIX WITH 0s
        self.phi = np.zeros((self.TOPICS, self.VOCABS))             # SPACE FOR PHI MATRIX WITH 0s

    def LoadData(self, record):                                     # FOR EACH RECORD
        if len(record) > 0:
            if self.header == True:
                self.header = False
            else:
                r = record.split("\t")                              # r[0] = DocID, r[1] = WordID, r[2] = Frequency
                tmp = [r[1] for i in range(int(r[2]))]
                if not r[0] in self.documents:                      # ADD DOCUMENT
                    self.documents[r[0]] = tmp
                    self.doc_to_index[r[0]] = self.DOCS
                    self.DOCS += 1
                else:
                    self.documents[r[0]] += tmp
                if not r[1] in self.word_to_index:                           # ADD WORD
                    self.word_to_index[r[1]] = self.VOCABS
                    self.VOCABS += 1

    def assignTopics(self, doc, word, pos):                         # DRAW TOPIC SAMPLE FROM FULL-CONDITIONAL DISTRIBUTION
        d = self.doc_to_index[doc]
        w = self.word_to_index[word]
        z = self.topicAssignments[d][pos]                           # TOPIC ASSIGNMENT OF WORDS FOR EACH DOCUMENT
        self.cntTW[z, w] -= 1
        self.cntDT[d, z] -= 1
        self.cntT[z] -= 1
        self.lenD[d] -= 1

        # FULL-CONDITIONAL DISTRIBUTION
        prL = (self.cntDT[d] + self.alpha) / (self.lenD[d] + np.sum(self.alpha))
        prR = (self.cntTW[:,w] + self.beta) / (self.cntT + self.beta * self.VOCABS)
        prFullCond = prL * prR                                      # FULL-CONDITIONAL DISTRIBUTION
        prFullCond /= np.sum(prFullCond)                            # TO OBTAIN PROBABILITY
        # NOTE: 'prFullCond' is MULTINOMIAL DISTRIBUTION WITH THE LENGTH, NUMBER OF TOPICS, NOT A VALUE
        new_z = np.random.multinomial(1, prFullCond).argmax()       # RANDOM SAMPLING FROM FULL-CONDITIONAL DISTRIBUTION
        self.topicAssignments[d][pos] = new_z
        self.cntTW[new_z, w] += 1
        self.cntDT[d, new_z] += 1
        self.cntT[new_z] += 1

    def LogLikelihood(self):                                        # FIND (JOINT) LOG-LIKELIHOOD VALUE
        l = 0
        for z in range(self.TOPICS):                                # log p(w|z,\beta)
            l += gammaln(self.VOCABS * self.beta)
            l -= self.VOCABS * gammaln(self.beta)
            l += np.sum(gammaln(self.cntTW[z] + self.beta))
            l -= gammaln(np.sum(self.cntTW[z] + self.beta))
        for doc in self.documents:                                  # log p(z|\alpha)
            d = self.doc_to_index[doc]
            l += gammaln(np.sum(self.alpha))
            l -= np.sum(gammaln(self.alpha))
            l += np.sum(gammaln(self.cntDT[d] + self.alpha))
            l -= gammaln(np.sum(self.cntDT[d] + self.alpha))
        return l

    def findThetaPhi(self):
        th = np.zeros((self.DOCS, self.TOPICS))                     # SPACE FOR THETA
        ph = np.zeros((self.TOPICS, self.VOCABS))                   # SPACE FOR PHI
        for d in range(self.DOCS):
            for z in range(self.TOPICS):
                th[d][z] = (self.cntDT[d][z] + self.alpha[z]) / (self.lenD[d] + np.sum(self.alpha))
        for z in range(self.TOPICS):
            for w in range(self.VOCABS):
                ph[z][w] = (self.cntTW[z][w] + self.beta) / (self.cntT[z] + self.beta * self.VOCABS)
        return ph, th

    def run(self, nsamples, burnin, interval):                   # GIBBS SAMPLER KERNEL
        if nsamples <= burnin:                                      # BURNIN CHECK
            print("ERROR: BURN-IN POINT EXCEEDS THE NUMBER OF SAMPLES")
            sys.exit(0)
        print("# of DOCS:", self.DOCS)                              # PRINT TRAINING DATA INFORMATION
        print("# of TOPICS:", self.TOPICS)
        print("# of VOCABS:", self.VOCABS)

        # MAKE SPACE FOR TOPIC-ASSIGNMENT MATRICES WITH 0s
        self.topicAssignments = {}                                  # {INDEX OF DOC: [TOPIC ASSIGNMENT]}
        for doc in self.documents:
            d = self.doc_to_index[doc]
            self.topicAssignments[d] = [0 for word in self.documents[doc]]
        self.cntTW = np.zeros((self.TOPICS, self.VOCABS))           # NUMBER OF TOPICS ASSIGNED TO A WORD
        self.cntDT = np.zeros((self.DOCS, self.TOPICS))             # NUMBER OF TOPICS ASSIGNED IN A DOCUMENT
        self.cntT = np.zeros(self.TOPICS)                           # ASSIGNMENT COUNT FOR EACH TOPIC
        self.lenD = np.zeros(self.DOCS)                             # ASSIGNMENT COUNT FOR EACH DOCUMENT = LENGTH OF DOCUMENT

        # RANDOMLY ASSIGN TOPIC TO EACH WORD
        for doc in self.documents:
            for i, word in enumerate(self.documents[doc]):
                d = self.doc_to_index[doc]
                w = self.word_to_index[word]
                rt = random.randint(0, self.TOPICS-1)               # RANDOM TOPIC ASSIGNMENT
                self.topicAssignments[d][i] = rt                    # RANDOMLY ASSIGN TOPIC TO EACH WORD
                self.cntTW[rt, w] += 1
                self.cntDT[d, rt] += 1
                self.cntT[rt] += 1
                self.lenD[d] += 1

        # COLLAPSED GIBBS SAMPLING
        print("INITIAL STATE")
        print("\tLikelihood:", self.LogLikelihood())               # FIND (JOINT) LOG-LIKELIHOOD
        print("\tAlpha:", end="")
        for i in range(self.TOPICS):
            print(" %.5f" % self.alpha[i], end="")
        print("\n\tBeta: %.5f" % self.beta)
        SAMPLES = 0
        for s in range(nsamples):
            for doc in self.documents:
                for i, word in enumerate(self.documents[doc]):
                    self.assignTopics(doc, word, i)                 # DRAW TOPIC SAMPLE FROM FULL-CONDITIONAL DISTRIBUTION
            lik = self.LogLikelihood()
            print("SAMPLE #" + str(s))
            print("\tLikelihood:", lik)
            print("\tAlpha:", end="")
            for i in range(self.TOPICS):
                print(" %.5f" % self.alpha[i], end="")
            print("\n\tBeta: %.5f" % self.beta)
            if s > burnin and s % interval == 0:                    # FIND PHI AND THETA AFTER BURN-IN POINT
                ph, th = self.findThetaPhi()
                self.theta += th
                self.phi += ph
                SAMPLES += 1
        self.theta /= SAMPLES                                       # AVERAGING GIBBS SAMPLES OF THETA
        self.phi /= SAMPLES                                         # AVERAGING GIBBS SAMPLES OF PHI
        return (theta, phi, lik)
