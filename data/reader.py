import os
import string
import tensorflow as tf
import tokenize as tk
from io import BytesIO
import pickle

def prepare_raw_data(force=False, test=False):
  raw_data_path = 'data/raw_data.pickle'
  if force == False and os.path.exists(raw_data_path):
    with open(raw_data_path, 'rb') as handle:
      raw_data = pickle.load(handle)
      return raw_data

  raw_data = []
  count = 0
  for root, dirs, files in os.walk('data/rails'):
    for file in files:
      if file.endswith('.rb'):
        file_path = os.path.join(root,file)
        print(file_path)
        with tf.gfile.GFile(file_path, "r") as f:
          text = f.read().decode("utf-8")
          raw_data.append('<bof>')
          for line in text.splitlines():
            indent_num = len(line) - len(line.lstrip())
            delimiter = '　' 
            line = add_delimiter(delimiter,line, string.punctuation)
            line = line.replace(' ',delimiter + ' ' + delimiter)
            tokens = line.split(delimiter)
            tokens = filter(None, tokens)
            raw_data.extend(tokens)
            raw_data.append('<nl>')
          raw_data.append('<eof>')
        if test==True:
          print(raw_data)
          return raw_data
  with open(raw_data_path, 'wb') as handle:
    print(len(raw_data))
    pickle.dump(raw_data, handle)
  return raw_data

def print_raw_data(raw_data):
  symbols = ':.",[]()_|<>@'
  for i, tok in enumerate(raw_data):
    if tok == '<bof>':
      continue
    elif tok == '<eof>':
      print("="*50)
    elif tok == '<nl>':
      print('')
    else:
      print(tok, end='')

def print_indent(indent):
  print(' ' * int(indent[1]), end='')

def add_delimiter(delimiter, text, symbols):
  """Add delimiter before and after a symbol"""
  for symbol in symbols:
    text = text.replace(symbol, "{}{}{}".format(delimiter,symbol,delimiter))
  return text

def build_dict(raw_data, force=False):
  dict_path = 'data/dictionary.pickle'
  if force == False and os.path.exists(dict_path):
    with open(dict_path, 'rb') as handle:
      dictionary = pickle.load(handle)
    return dictionary

  dictionary = dict({'<pad>': 0})
  for token in raw_data:
    if not token in dictionary:
      dictionary[token] = len(dictionary)
  print(len(dictionary))
  with open(dict_path, 'wb') as handle:
    pickle.dump(dictionary, handle)
  return dictionary

def convert_data(raw_data, dictionary, force=False):
  data_path = 'data/data.pickle'
  if force == False and os.path.exists(data_path):
    with open(data_path, 'rb') as handle:
      data = pickle.load(handle)
    return data

  data = []
  for token in raw_data:
    data.append(dictionary[token])
  with open(data_path, 'wb') as handle:
    pickle.dump(data, handle)
  return data

def print_data(data, dictionary):
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  raw_data = []
  for id in data:
    raw_data.append(reverse_dictionary[id])
  print_raw_data(raw_data)

def save(data)

if __name__ == '__main__':
  raw_data = prepare_raw_data(force=True, test=False)
  #print_raw_data(raw_data)
  dictionary = build_dict(raw_data, force=True)
  data = convert_data(raw_data, dictionary, force=True)
  print(data[:100])
  print_data(data[:1000], dictionary)

  