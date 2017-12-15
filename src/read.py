import xml.etree.ElementTree as et
import os
import re
from unidecode import unidecode
import optparse
import os.path
import subprocess
from multiprocessing import Pool

class OpenNLP():
    def __init__(self, tool, path_opennlp, path_model):
        self.tool = tool
        self.path_opennlp = path_opennlp
        self.path_model = path_model
        # spawn the process

    def parse(self, text):
        self.process = subprocess.Popen([self.path_opennlp, self.tool, self.path_model], stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.PIPE)
        response = self.process.communicate(input=((text).encode()))
        self.process.wait()
        # print(response)
        sentences = response[0].decode().strip().split('\n')

        return sentences

def preprocess(content):
    sent_parser = OpenNLP("SentenceDetector", "../apache-opennlp-1.8.3/bin/opennlp", "models/en-sent.bin")
    token_parser = OpenNLP("TokenizerME", "../apache-opennlp-1.8.3/bin/opennlp", "models/en-token.bin")

    content = unidecode(content)
    content = content.strip()
    urls = ""
    urlregex = re.compile(r'((http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    for match in urlregex.finditer(content):
        start, end = match.span()
        url = content[start:end]
        splitregex = re.compile(r'\W+')
        words = splitregex.split(url)
        processedurl = ' '.join(words)
        # content = content[:start] + processedurl + content[end:]
        urls += " "
        urls += processedurl

    content = re.sub(urlregex, '', content)
    content = content + urls
    content = sent_parser.parse(content)
    final_content = ""
    for sentence in content:
        sentence = sentence.lower()
        sentence = token_parser.parse(sentence)
        final_content += sentence[0]
        final_content += " "

    return final_content

def map_function(filename):

    author_name = re.search('\/(\d+)\.', filename).groups(0)[0]
    print(author_name)

    next_doc_id = 0

    f = open(filename, 'r', encoding='iso-8859-1')
    contents = f.read()
    seek1 = contents.find('<post>')
    seek2 = contents.find('</post>', seek1+1)
    while(seek1!=-1):
        print("\t", next_doc_id)
        post = contents[seek1+6:seek2]
        seek1 = contents.find('<post>', seek1+1)
        seek2 = contents.find('</post>', seek1+1)

        post = preprocess(post)


        outfile = open(destination_path + '/' + author_name + "-" + str(next_doc_id) + '.txt', 'w')
        outfile.write(post)
        outfile.close()

        next_doc_id += 1

    f.close()

def readblogs(source_path):
    filenames = [source_path + file for file in os.listdir(source_path)]
    with Pool(32) as p:
        p.map(map_function, filenames)

def readjudgment(source_path):
    filenames = [source_path + file for file in os.listdir(source_path)]
    with Pool(32) as p:
        p.map(map_judgment, filenames)

def map_judgment(filename):
    quotes = re.compile(r'\".*?\"')
    numbers = re.compile(r'\d+')
    fileregex = re.compile(r'(\w+)(\d+)')

    f = open(filename, 'r')
    contents = f.read()
    contents = re.sub(quotes, "", contents)
    contents = re.sub(numbers, "", contents)
    print("preprocess now")
    contents = preprocess(contents)
    print("preprocess done")
    new_filename = re.sub(fileregex, '\1\-\2\.txt', filename)
    outfile = open(destination_path + new_filename, 'w')
    outfile.write(content)
    outfile.close()
    print(new_filename)

def readpan11(source_path):
    filenames = os.listdir(source_path)
    with Pool(32) as p:
        p.starmap(map_pan11_test, zip(filenames, repeat(source_path+'training/')))

def map_pan11_test(author, source):
    numbers_regex = r'\d+'
    filenames = os.listdir(source_path)
    candidate_number = re.search(numbers_regex, author)
    for filename in filenames:
        file_number = re.search(numbers_regex, filename)
        f = open(filename, 'r')
        contents = f.read()
        contents = preprocess(contents)
        outfile = open(destination_path + 'test/' + new_filename, 'w')
        outfile.write(content)
        outfile.close()

# destination_path = '../data/judgmentprocessed'
destination_path = '../data/pan11processed/'
# readblogs('../data/blogs/')
readjudgment('../data/Judgment/used/')
