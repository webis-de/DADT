#!/usr/bin/
import os
import re
import optparse
import os.path
import subprocess
from multiprocessing import Pool

source_path = "../data/blogsprocessed/"
# source_path = "../data/nichtschiller/"

filenames = [source_path + file for file in os.listdir(source_path)]

author_pattern = re.compile(r'\/(\w+)-\d+\.txt')

author_counts = {}

for filename in filenames:
    match = author_pattern.search(filename)
    author = match.group(1)
    if author not in author_counts:
        author_counts[author] = 0
    author_counts[author] += 1

total_documents = 0
prolific_authors = set()
i = 0
for author, freq in sorted(author_counts.items(), key=lambda x:x[1], reverse = True):
    if i < 1000:
        print(author,i,freq)
        prolific_authors.add(author)
        total_documents += freq
        i += 1

target = open("smallfiles.txt","w")

for filename in filenames:
    match = author_pattern.search(filename)
    author = match.group(1)
    if author in prolific_authors:
        target.write(filename + "\n")

target.close()

print("All authors wrote a total of",total_documents,"documents")
