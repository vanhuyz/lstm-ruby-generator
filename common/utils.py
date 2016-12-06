import pickle

def load_dictionary():
  with open('data/dicts/dictionary.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)
  return dictionary

def load_raw_data():
  with open('data/dicts/raw_data.pickle', 'rb') as handle:
    raw_data = pickle.load(handle)
  return raw_data

def load_data():
  with open('data/dicts/data.pickle', 'rb') as handle:
    data = pickle.load(handle)
  return data

def print_raw_data(raw_data):
  for i, tok in enumerate(raw_data):
    if tok == '<bof>' or tok == '<pad>':
      continue
    elif tok == '<eof>':
      print("="*50)
    elif tok == '<nl>':
      print('')
    else:
      print(tok, end='')
  print("n")

def print_data(data):
  dictionary = load_dictionary()
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  raw_data = []
  raw_data.extend(reverse_dictionary[id] for id in data)
  #print_raw_data(raw_data)
  print(''.join(raw_data))