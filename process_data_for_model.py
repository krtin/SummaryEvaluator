import pickle as pkl
import pandas as pd
import numpy as np
import collections
import struct
import tensorflow as tf
from tensorflow.core.example import example_pb2
import os

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
VOCAB_SIZE = 200000

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

smic_train = "../../rougetrick_clean/data/discriminator_corpus_train.pkl"
smic_val = "../../rougetrick_clean/data/discriminator_corpus_val.pkl"
smic_test = "../../rougetrick_clean/data/discriminator_corpus_test.pkl"
all_smics = [smic_test, smic_val, smic_train]

smc_train = "data/train.csv"
smc_val = "data/val.csv"
smc_test = "data/test.csv"
all_smcs = [smc_test, smc_val, smc_train]
outfiles = ["data/test_1.bin", "data/val_1.bin", "data/train_1.bin"]
makevocabs = [False, False, True]

error_counts = 0
total_data=0
#loop through all smic corpuses created through rougetrick
for smic_path, smc_path, outfile, makevocab in zip(all_smics, all_smcs, outfiles, makevocabs):
    #if(outfile=="data/test_1.bin" or outfile=="data/val_1.bin"):
    #    continue
    print("Starting to write file %s"%outfile)
    [smics, starting_batch, totaldropped] = pkl.load(open(smic_path, 'rb'))
    smcs = pd.read_csv(smc_path, dtype={"idx": np.int64, 'source': object, 'smc': object, })
    totalsmcs = len(smcs)
    writer = open(outfile, 'wb')


    if makevocab:
      vocab_counter = collections.Counter()

    for i, row in smcs.iterrows():
        sourceid = row['idx']
        smc = row['smc']
        source = row['source']

        #write smc to file
        score="2"
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


        for j, smic_row in smics[smics['sourceid']==sourceid].iterrows():
            if(np.random.rand(1)[0]>0.02):
                continue
            smic = smic_row['smic']
            ruleid = smic_row['ruleid']
            score="1"
            if(ruleid<5):
                score = "0"

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

        print("\rCompleted %d out of %d, Total Gen: %d"%(i, totalsmcs, total_data), end='')

    writer.close()
    print("Finished writing file %s"%outfile)

    # write vocab to file
    if makevocab:
      print("Writing vocab file...")
      with open(os.path.join("data", "vocab_1"), 'w') as writer:
        for word, count in vocab_counter.most_common(VOCAB_SIZE):
          writer.write(word + ' ' + str(count) + '\n')
      print("Finished writing vocab file")
