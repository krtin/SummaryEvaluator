import pickle as pkl
import pandas as pd
import numpy as np
import collections
import struct
import tensorflow as tf
from tensorflow.core.example import example_pb2
import os
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
VOCAB_SIZE = 200000

def tokenize(sents):
    parse = nlp.annotate(sents, properties={
      'annotators': 'tokenize,ssplit',
      'outputFormat': 'json'
    })

    sents = []
    for sent in parse["sentences"]:
        tokens = sent["tokens"]
        for token in tokens:

            sents.append(token["word"])
    sents = " ".join(sents)

    return sents

def get_bin_data(article, abstract, target):
    if(type(article)==float or type(abstract)==float):
        return None, None
    article = article.encode('utf-8')
    abstract = abstract.encode('utf-8')
    target = target.encode('utf-8')
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([article])
    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
    tf_example.features.feature['label'].bytes_list.value.extend([target])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)

    length = struct.pack('q', str_len)
    data = struct.pack('%ds' % str_len, tf_example_str)

    return length, data

smc_final_path = "../../rougetrick_clean/data/smc_final.csv"
smic_final_path = "../../rougetrick_clean/data/smic_final_new.csv"

smc_data = pd.read_csv(smc_final_path, dtype={'sourceid': object, 'judgeid': object, 'system': object, 'source': object}, usecols=['sourceid', 'judgeid', 'system', 'source'])

smic_data = pd.read_csv(smic_final_path, dtype={"judgeid": object, 'smic': object, 'sourceid': object, 'smic_id': np.int32}, usecols=['judgeid', 'smic', 'sourceid', 'smic_id'])
print(len(smic_data))
#combined_data = pd.merge(smic_data, smc_data, left_on=['sourceid', 'judgeid'], right_on=['sourceid', 'judgeid'], how='left')

#if(len(combined_data)!=len(smic_data)):
#    raise Exception("Some problem while merging")

sourceids_unique = list(smc_data['sourceid'].unique())
sourceids_len = len(sourceids_unique)

train_len = int(sourceids_len*0.8)
val_len = int(sourceids_len*0.1)
test_len = sourceids_len - train_len - val_len


sourceids_train = np.random.choice(sourceids_unique, train_len, replace=False)
sourceids_val = np.random.choice(list(set(sourceids_unique).difference(set(sourceids_train))), val_len, replace=False)
sourceids_test = list(set(sourceids_unique).difference(set(sourceids_train)).difference(set(sourceids_val)))

print(len(sourceids_train), len(sourceids_val), len(sourceids_test), len(sourceids_unique))
print((set(sourceids_train).union(set(sourceids_val), set(sourceids_test)))==set(sourceids_unique))

data_list = [sourceids_test, sourceids_val, sourceids_train]
outfiles = ["data/test_tout.bin", "data/val_tout.bin", "data/train_tout.bin"]
makevocabs = [False, False, True]


error_counts = 0
total_data=0

#loop through all smic corpuses created through rougetrick
for sourceids, outfile, makevocab in zip(data_list, outfiles, makevocabs):
    total_data=0
    counter = 0
    #if(outfile=="data/test_1.bin" or outfile=="data/val_1.bin"):
    #    continue
    print("Starting to write file %s"%outfile)

    smcs = smc_data[smc_data['sourceid'].isin(sourceids)]
    totalsmcs = len(smcs)
    print('%d SMCS found'%(totalsmcs))

    writer = open(outfile, 'wb')

    smics_set = smic_data[smic_data['sourceid'].isin(sourceids)]

    if makevocab:
      vocab_counter = collections.Counter()

    for i, row in smcs.iterrows():
        sourceid = row['sourceid']
        judgeid = row['judgeid']
        source = tokenize(row['source']).lower()
        smc = tokenize(row['system']).lower()
        smics = smics_set[smics_set['sourceid']==sourceid]
        smics = smics[smics['judgeid']==judgeid]
        #print(sourceid)
        #print(judgeid)
        #print(smics)
        #write smc to file
        length, data = get_bin_data(source, smc, "1")
        if(length==None):
            continue
        writer.write(length)
        writer.write(data)
        total_data+=1

        # Write the vocab to file, if applicable
        if makevocab:
          art_tokens = source.split(' ')
          abs_tokens = smc.split(' ')
          abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
          tokens = art_tokens + abs_tokens
          tokens = [t.strip() for t in tokens] # strip
          tokens = [t for t in tokens if t!=""] # remove empty
          vocab_counter.update(tokens)

        for j, smic_row in smics.iterrows():

            smic = smic_row['smic']
            #print(smic)

            score="1"


            if(smc==smic):
                error_counts+=1
                print("SMC and SMIC are the same total cases %d"%error_counts)
                continue
            #write smic to file
            length, data = get_bin_data(source, smic, "0")
            if(length==None):
                continue
            writer.write(length)
            writer.write(data)
            total_data+=1

        counter += 1
        print("\rCompleted %d out of %d, Total Gen: %d"%(counter, totalsmcs, total_data), end='')


    writer.close()
    print("Finished writing file %s"%outfile)

    # write vocab to file
    if makevocab:
      print("Writing vocab file...")
      with open(os.path.join("data", "vocab_tout"), 'w') as writer:
        for word, count in vocab_counter.most_common(VOCAB_SIZE):
          writer.write(word + ' ' + str(count) + '\n')
      print("Finished writing vocab file")
