import os
import codecs
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
from util.data_util import load_json, load_lines, load_pickle, save_pickle, time_to_index

PAD, UNK = "<PAD>", "<UNK>"


class Processor:
    def __init__(self):
        super(Processor, self).__init__()
        self.idx_counter = 0

    def reset_idx_counter(self):
        self.idx_counter = 0

    def process_data(self, data, scope):
        results = []
        for record in tqdm(data, total=len(data), desc='process data {}'.format(scope)):
            vid, duration, gt_label, sentence = record[:4]
            start_time, end_time = gt_label
            words = word_tokenize(sentence.strip().lower(), language="english")

            record = {'sample_id': self.idx_counter, 'vid': str(vid), 's_time': start_time, 'e_time': end_time,
                    'duration': duration, 'words': words}
            results.append(record)
            self.idx_counter += 1
        return results

    def convert(self, data_dir):
        self.reset_idx_counter()
        if not os.path.exists(data_dir):
            raise ValueError('data dir {} does not exist'.format(data_dir))
        # load raw data
        train_data = load_json(os.path.join(data_dir, 'train.json'))
        test_data = load_json(os.path.join(data_dir, 'test.json'))

        # process data
        train_set = self.process_data(train_data, scope='train')
        test_set = self.process_data(test_data, scope='test')
        return train_set, None, test_set  # train/val/test


def load_glove(glove_path):
    vocab = list()
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            vocab.append(word)
    return set(vocab)


def filter_glove_embedding(word_dict, glove_path):
    vectors = np.zeros(shape=[len(word_dict), 300], dtype=np.float32)
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                vectors[word_index] = np.asarray(vector)
    return np.asarray(vectors)


def vocab_emb_gen(datasets, emb_path):
    # generate word dict and vectors
    emb_vocab = load_glove(emb_path)
    word_counter, char_counter = Counter(), Counter()
    for data in datasets:
        for record in data:
            for word in record['words']:
                word_counter[word] += 1
                for char in list(word):
                    char_counter[char] += 1
    word_vocab = list()
    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)
    tmp_word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    vectors = filter_glove_embedding(tmp_word_dict, emb_path)
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    # generate character dict
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= 5]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])

    
    # charades
    # char_dict = {'<PAD>': 0,  '<UNK>': 1,  'o': 2,  'e': 3,  'n': 4,  's': 5,  't': 6,  'r': 7,  'a': 8, 
    # 'p': 9,  'i': 10,  'h': 11,  '.': 12,  'g': 13,  'l': 14,  'd': 15,  'u': 16,  'c': 17,
    # 'w': 18,  'f': 19,  'k': 20,  'b': 21,  'm': 22,  'y': 23,  'v': 24,  'x': 25,  'z': 26,
    # 'j': 27,  ',': 28,  "'": 29,  'q': 30,  '/': 31,  '-': 32,  '2': 33,  '#': 34,  '1': 35}

    # activity
    # char_dict = {'<PAD>': 0, '<UNK>': 1, 'e': 2, 'a': 3, 'n': 4, 't': 5, 'o': 6, 's': 7, 'i': 8, 'h': 9, 'r': 10,
    #     'l': 11, 'd': 12, 'g': 13, 'p': 14, 'm': 15, 'c': 16, 'w': 17, 'u': 18, '.': 19, 'f': 20, 'b': 21,
    #     'k': 22, 'y': 23, 'v': 24, ',': 25, 'j': 26, 'x': 27, "'": 28, 'q': 29, '`': 30, 'z': 31, '-': 32,
    #     '1': 33, '0': 34, '2': 35, '3': 36, '4': 37, '5': 38, '6': 39, '9': 40, '8': 41, '7': 42, 'ñ': 43,
    #     ':': 44, '/': 45, '!': 46, '&': 47, ')': 48, '(': 49, '?': 50, '#': 51, 'é': 52, '’': 53, ';': 54, '@': 55}

    # tacos
    # char_dict = {'<PAD>': 0, '<UNK>': 1, 'e': 2, 't': 3, 'o': 4, 's': 5, 'h': 6, 'a': 7, 'n': 8, 'r': 9, 'i': 10,
    #             'p': 11, 'l': 12, 'd': 13, 'c': 14, 'u': 15, 'f': 16, '.': 17, 'g': 18, 'w': 19, 'm': 20,
    #             'k': 21, 'b': 22, 'v': 23, 'y': 24, ',': 25, 'j': 26, 'x': 27, 'q': 28, 'z': 29, '-': 30,
    #             "'": 31, '2': 32, '/': 33, '1': 34, '(': 35, ')': 36, '4': 37, '?': 38, ':': 39, '`': 40,
    #             '3': 41, '!': 42, ';': 43, '#': 44}

    return word_dict, char_dict, vectors


def dataset_gen(data, vfeat_lens, word_dict, char_dict, max_pos_len, scope):
    dataset = list()
    for record in tqdm(data, total=len(data), desc='process {} data'.format(scope)):
        vid = record['vid']
        if vid not in vfeat_lens:
            continue
        s_ind, e_ind, _ = time_to_index(record['s_time'], record['e_time'], vfeat_lens[vid], record['duration'])
        word_ids, char_ids = [], []
        for word in record['words'][0:max_pos_len]:
            word_id = word_dict[word] if word in word_dict else word_dict[UNK]
            char_id = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_ids.append(word_id)
            char_ids.append(char_id)
        result = {'sample_id': record['sample_id'], 'vid': record['vid'], 's_time': record['s_time'],
                  'e_time': record['e_time'], 'duration': record['duration'], 'words': record['words'],
                  's_ind': int(s_ind), 'e_ind': int(e_ind), 'v_len': vfeat_lens[vid], 'w_ids': word_ids,
                  'c_ids': char_ids}
        dataset.append(result)
    return dataset


def gen_or_load_dataset(configs):
    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)
    
    data_dir = os.path.join('data', configs.task + "_" + configs.suffix)
    feature_dir = configs.feature_dir


    if configs.suffix is None:
        save_path = os.path.join(configs.save_dir, '_'.join([configs.task, str(configs.max_pos_len)]) +
                                 '.pkl')
    else:
        save_path = os.path.join(configs.save_dir, '_'.join([configs.task, str(configs.max_pos_len),
                                                             configs.suffix]) + '.pkl')
    if os.path.exists(save_path):
        dataset = load_pickle(save_path)
        return dataset
    feat_len_path = os.path.join(feature_dir, 'feature_shapes.json')
    emb_path = configs.emb_path 
    # load video feature length
    vfeat_lens = load_json(feat_len_path)
    for vid, vfeat_len in vfeat_lens.items():
        vfeat_lens[vid] = min(configs.max_pos_len, vfeat_len)
    # load data
    processor = Processor()
    
    train_data, val_data, test_data = processor.convert(data_dir)
    # generate dataset
    data_list = [train_data, test_data] if val_data is None else [train_data, val_data, test_data]
    word_dict, char_dict, vectors = vocab_emb_gen(data_list, emb_path)
    train_set = dataset_gen(train_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'train')
    val_set = None if val_data is None else dataset_gen(val_data, vfeat_lens, word_dict, char_dict,
                                                        configs.max_pos_len, 'val')
    test_set = dataset_gen(test_data, vfeat_lens, word_dict, char_dict, configs.max_pos_len, 'test')
    # save dataset
    n_val = 0 if val_set is None else len(val_set)
    dataset = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set, 'word_dict': word_dict,
               'char_dict': char_dict, 'word_vector': vectors, 'n_train': len(train_set), 'n_val': n_val,
               'n_test': len(test_set), 'n_words': len(word_dict), 'n_chars': len(char_dict)}
    save_pickle(dataset, save_path)
    return dataset
