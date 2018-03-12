import xml.etree.ElementTree as et
import os
import re
from unidecode import unidecode
import optparse
import os.path
import subprocess
from multiprocessing import Pool
import json
from itertools import repeat

class OpenNLP():
    def __init__(self, tool, opennlp_dir, path_model):
        jars = subprocess.check_output(['bash', '-c' , 'echo ' + opennlp_dir + '/lib/*.jar ' + opennlp_dir + "/output/*.jar | tr ' ' ':'"]).decode(encoding="utf-8").strip()
        cp = opennlp_dir + "/output/classes:" + jars
        java = subprocess.check_output(['bash', '-c', 'echo $JAVA_HOME']).decode(encoding="utf-8").strip() + "/bin/java"

        self.command = [java,"-Xmx1000m","-classpath",cp,"-Dopennlp.dir=" + opennlp_dir, tool, path_model]



        # spawn the process

    def parse(self, text):
        self.process = subprocess.Popen(self.command, stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.PIPE)
        response = self.process.communicate(input=((text).encode()))
        self.process.wait()
        # print(response)
        sentences = response[0].decode().strip().split('\n')

        return sentences

def preprocess(content):
    sent_parser = OpenNLP("opennlp.tools.lang.english.SentenceDetector", "../opennlp-tools-1.4.3", "../opennlp-models/EnglishSD.bin.gz")
    token_parser = OpenNLP("opennlp.tools.lang.english.Tokenizer", "../opennlp-tools-1.4.3", "../opennlp-models/EnglishTok.bin.gz")

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
    contents = preprocess(contents)
    new_filename = re.sub(fileregex, '\1\-\2\.txt', filename)
    outfile = open(destination_path + new_filename, 'w')
    outfile.write(content)
    outfile.close()
    print(new_filename)

def readpan11(source_path):
    filenames = os.listdir(source_path+'training/')
    with Pool(32) as p:
        p.starmap(map_pan11_train, zip(list(filenames), repeat(source_path+'training/')))

    ground_file = open(source_path + 'ground-truth.json','r')
    ground_string = ground_file.read()
    ground_file.close()
    ground_json = json.loads(ground_string)
    ground_dict = ground_json['ground-truth']

    with Pool(32) as p:
        p.starmap(map_pan11_test, zip(ground_dict, repeat(source_path)))

def map_pan11_test(info, source_path):
    numbers_regex = r'\d+'
    author = info["true-author"]
    filename = info["unknown-text"]
    file = open(source_path + "unknown/" + info["unknown-text"], "r")
    candidate_number = re.search(numbers_regex, author).group()
    file_number = re.search(numbers_regex, filename).group()
    contents = file.read()
    file.close()
    contents = preprocess(contents)
    print("true author", candidate_number)
    print("unknown text", file_number)
    outfile = open(destination_path + 'test/' + candidate_number + '-' + file_number + '.txt', 'w')
    outfile.write(contents)
    outfile.close()

def map_pan11_train(author, source):
    numbers_regex = r'\d+'
    filenames = os.listdir(source+author)
    candidate_number = re.search(numbers_regex, author).group()
    for filename in filenames:
        file_number = re.search(numbers_regex, filename).group()
        print("BEGIN" , author + "/" + filename)
        f = open(source + author + "/" + filename, 'r')
        contents = f.read()
        contents = preprocess(contents)
        outfile = open(destination_path + 'train/' + candidate_number + '-' + file_number + '.txt', 'w')
        outfile.write(contents)
        outfile.close()
        print("END" , author + "/" + filename)

# destination_path = '../data/judgmentprocessed'
destination_path = '../data/pan11processed_new/'
# readblogs('../data/blogs/')
# readjudgment('../data/Judgment/used/')
# readpan11('../data/c10/')
readpan11('../data/pan11/')
