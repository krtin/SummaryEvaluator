import os
import struct

CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data
chunks_dir = os.path.join("data", "chunked")

def chunk_file(set_name):
  in_file = 'data/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(os.path.join(chunks_dir, set_name), '%s_%04d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1
      print("\rCompleted %d chunks"%(chunk), end='')

def chunk_all(nm=None):
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train_tout', 'val_tout', 'test_tout']:
    if(set_name!=nm and nm!=None):
        continue
    if not os.path.isdir(os.path.join(chunks_dir, set_name)):
      os.mkdir(os.path.join(chunks_dir, set_name))
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)

chunk_all(nm=None)
