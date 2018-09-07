import numpy as np
import pandas as pd
import os
import struct
import subprocess
import tensorflow as tf
from tensorflow.core.example import example_pb2

toutanova_corpus_raw = "toutanova_corpus/"

list_of_files = [f for f in os.listdir(toutanova_corpus_raw) if os.path.isfile(os.path.join(toutanova_corpus_raw, f))]

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

def fix_missing_period(line):

  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."

def get_bin_data(source, summary):

  mapping_file = os.path.join(toutanova_corpus_raw, "mapping.txt")
  tmp_inputfile = os.path.join(toutanova_corpus_raw, "tmp_input.txt")
  tmp_outputfile = os.path.join(toutanova_corpus_raw, "tmp_output.txt")

  with open(tmp_inputfile, 'w') as f:
      f.write("%s\n%s"%(source, summary))

  with open(mapping_file, "w") as f:
      f.write("%s \t %s\n" % (tmp_inputfile, tmp_outputfile))


  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', mapping_file]

  subprocess.call(command)
  os.remove(tmp_inputfile)
  if(os.path.exists(tmp_outputfile)):
      with open(tmp_outputfile, 'r') as f:
          text = f.read().split('\n')
          if(len(text)!=2):
              raise Exception("Error reading tokenized file, length of file is %d"%(len(text)))
          source_tok = text[0]
          summary_tok = text[1]
      os.remove(tmp_outputfile)

  else:
      raise Exception("Error during tokenization")

  source_tok = source_tok.lower()
  summary_tok = summary_tok.lower()
  source_tok = fix_missing_period(source_tok)
  summary_tok = fix_missing_period(summary_tok)
  source_tok = source_tok.encode('utf-8')
  summary_tok = summary_tok.encode('utf-8')
  label = "0".encode('utf-8')
  tf_example = example_pb2.Example()
  tf_example.features.feature['article'].bytes_list.value.extend([source_tok])
  tf_example.features.feature['abstract'].bytes_list.value.extend([summary_tok])
  #just dummy
  tf_example.features.feature['label'].bytes_list.value.extend([label])
  tf_example_str = tf_example.SerializeToString()
  str_len = len(tf_example_str)

  return str_len, tf_example_str


for filename in list_of_files:
    if("test" not in filename):
        continue
    filepath = os.path.join(toutanova_corpus_raw, filename)
    datatype = filename.split("_")
    if(len(datatype)<2):
        datatype = "test"
        continue
    else:
        datatype = datatype[1].split(".")[1]

    output_file = os.path.join("correlation", datatype+".bin")
    writer = open(output_file, 'wb')
    print("Starting to write to %s"%output_file)
    error = 0
    count= 0
    with open(filepath, 'r') as f:
        data = f.read().split('\n')
        for row in data:
            if(row==''):
                continue
            row = row.split('|||')
            if(len(row)!=2):
                #print("Compression info does not exists")
                #print(row[0])
                error+=1
                continue

            sourceinfo = row[0]
            compressioninfo = row[1]

            [sourceid, domain, source] = sourceinfo.split('\t')
            #print(sourceid)
            #print(domain)
            #print(source)
            compressioninfo = compressioninfo.split('\t')
            if(len(compressioninfo)<3):
                raise Exception("Error processing compression info")
            summary = compressioninfo[0]
            judgeid = compressioninfo[1]
            noof_ratings = int(compressioninfo[2])
            #print(summary)
            #print(judgeid)
            #print(noof_ratings)
            str_len, tf_example_str = get_bin_data(source, summary)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))
            count += 1
            humanratings = compressioninfo[3:]
            #for humanrating in humanratings:
            #    print(humanrating)
    print("Finished writing to %s with error count %d and data count %d"%(output_file, error, count))
    writer.close()
