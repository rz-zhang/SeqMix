import argparse
import csv
import json
import logging
import os
import random
import sys
import copy
import time
import math
import numpy as np
import torch
import torch.nn.functional as F

from prettytable import PrettyTable
from torch.autograd import Variable
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from seqeval.metrics import classification_report

from model import Ner
from data_load import readfile, NerProcessor, convert_examples_to_features
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from active_learn import nte_sampling, random_sampling, uncertainty_sampling

'''Get Word Embedding'''
def get_word_embedding(sp_output_dir=None):
  model = Ner.from_pretrained(sp_output_dir)
  tokenizer = BertTokenizer.from_pretrained(sp_output_dir, do_lower_case=args.do_lower_case)

  for name, parameters in model.named_parameters():
    #print(name,':',parameters.size())
    if name=='bert.embeddings.word_embeddings.weight':
      bert_embedding = parameters.detach().cpu().numpy()

  wordidx2ebd = {idx:bert_embedding[idx] for idx in range(bert_embedding.shape[0])}
  ebd2wordidx = {}
  for k,v in wordidx2ebd.items():
    ebd2wordidx[tuple(v)] = k

  return wordidx2ebd, ebd2wordidx

def x_reconstruct(sequence):
  '''
  Experiment helper function. To reconstruct the sentence from a series of idx of word
  '''
  seq_list = []
  seq_str = ' '
  for item in sequence:
    if item == n_words-1:
      break
    seq_list.append(idx2word[item])
  return seq_str.join(seq_list)

def score_gpt(sentence):
  tokenize_input = gpt_tokenizer.tokenize(sentence)
  tensor_input = torch.tensor([gpt_tokenizer.convert_tokens_to_ids(tokenize_input)])
  loss=gpt_model(tensor_input, lm_labels=tensor_input)
  return math.exp(loss)

def high_quality(sequence, score_limit_upper=500, score_limit_low=0):
  score = score_gpt(sequence)
  return (score>=score_limit_low and score<score_limit_upper), score

"""## Utility function"""

def extract_same_elem(list1, list2):
  '''
  Utility function to extract the same element in two list, will be called in function find_subseq_pair
  '''
  set1 = set(list(list1))
  set2 = set(list(list2))
  iset = set1.intersection(set2)
  return list(iset)


def most_similar(value, top_n):
  '''
  Get the top_n most similar words given an embedding

  Input:
    value: a word embedding
    top_n: the number of entries(candidates) in the returned array
  Output:
    words_array: candidate array which contains the top_n most similar words, each entry is a word
  '''
  distance = np.sum(np.square(value - mydict_values), axis=1)
  idx = multi_argmax(-distance,top_n)
  token_pool = []
  for item in idx:
    token_pool.append(tokenizer.convert_ids_to_tokens(int(item)))
  return token_pool


def find_sub_seq(sequence, window_size, valid_tag_bar):
  '''
  Find the sub-sequence given a window_size and the limit of valid tags.

  Input:
    sequence: a single sequence with the shape (max_len,)
    window_size: the length of sub-sequence 
    valid_tag_bar: the lower limit over which the sub-sequence will be consider valid
  Output:
    sub_sequence: the concret sub-sequence represented by the idx of tag
    subseq_start_index: the list of starting index
  '''
  sub_sequence = []
  subseq_start_index = []
  for index in range(0,len(sequence)-window_size):
    valid_tag_count = 0
    exclude_label = ['O','[CLS]','[SEP]','[UKN]']
    if sequence[index] not in exclude_label:
      for item in sequence[index: index+window_size]:
        if item not in exclude_label:
          valid_tag_count += 1
      if valid_tag_count >= valid_tag_bar:
        sub_sequence.append(tuple(sequence[index: index+window_size]))
        subseq_start_index.append(index)
  return sub_sequence, subseq_start_index

def soft_pair(candidate):
  # valid tag num:
  valid_tag_list = []
  for idx, item in enumerate(candidate):
    tags = item.label
    exclude_label = ['O','[CLS]','[SEP]','[UKN]']
    valid_tag_count = 0
    for tag in tags:
      if tag not in exclude_label:
        valid_tag_count += 1
    if valid_tag_count >= valid_tag_bar:
      valid_tag_list.append(idx)
  # only search in the sentences with enough valid tags
  valid_len_list = []
  for item in candidate[valid_tag_list]:
    valid_len_list.append(len(item.label))
  # equal length index list (52,)
  equal_len_index_list = []
  for length in range(args.max_seq_length):
    equal_len_index = np.where(np.array(valid_len_list)==length+1)
    equal_len_index_list.append(np.array(valid_tag_list)[list(equal_len_index)])
  return equal_len_index_list

def soft_pair_index_generator(equal_len_index_list, pair_num, valid_tag_bar):
  pair_index_list = []
  len_range = []
  for i in range(len(equal_len_index_list)):
    if equal_len_index_list[i].shape[0]>=2:
      len_range.append(i+1)
  for i in range(pair_num):
    temp_len = random.choice(len_range)
    pair_index_list.append(random.sample(list(equal_len_index_list[temp_len-1]),2))
  return pair_index_list


def lf_mixup(mixdata, sentence1_idx, start_idx1, sentence2_idx, start_idx2, hyper_lambda):
  '''
  Label-fixed Mixup.
  Note that here the start_idx1 and start_idx2 are only int rather than array.
  '''
  new_seq1 = list((mixdata[sentence1_idx].text_a).split())
  new_seq2 = list((mixdata[sentence2_idx].text_a).split())

  mix_seq = []
  # mixup
  for i in range(mix_len):
    e1 = wordidx2ebd[tokenizer.convert_tokens_to_ids(new_seq1[start_idx1+i])]
    e2 = wordidx2ebd[tokenizer.convert_tokens_to_ids(new_seq2[start_idx2+i])]
    e_mix = hyper_lambda*e1 + (1-hyper_lambda)*e2
    mix_token = most_similar(e_mix, 7) # return candidate word pool
    exclude_pool = [new_seq1[start_idx1+i], new_seq2[start_idx2+i], '[UNK]', '[CLS]', '[SEP]', '[PAD]']
    for token in mix_token:
      if token not in exclude_pool and token.find('[unused')==-1 and token.find('##')==-1 :
        mix_seq.append(str(token))
        #print(token)
        break

  # substitution
  for i in range(mix_len):
    try:
      new_seq1[start_idx1+i] = mix_seq[i]
      new_seq2[start_idx2+i] = mix_seq[i]
    except:
      print('\n---NEW SEQ 1 - LENGTH = {}, START IDX = {}---\n'.format(len(new_seq1), start_idx1))
      print('\n---NEW SEQ 2 - LENGTH = {}, START IDX = {}---\n'.format(len(new_seq2), start_idx2))
      continue
  new_seq_1 = ' '.join(new_seq1)
  new_seq_2 = ' '.join(new_seq2)
  new_seq1_tag = mixdata[sentence1_idx].label
  new_seq2_tag = mixdata[sentence2_idx].label

  return new_seq_1, new_seq_2, new_seq1_tag, new_seq2_tag


def lf_augment(candidate_data, num_mixup, hyper_alpha, score_limit_upper=500, score_limit_low=0):
  '''
  Label_fixed augment.
  Given the candidate dataset and number of samples to be generated via mixup method, augment the training dataset by implementing mixup.
  '''
  global GUID_COUNT
  time_start=time.time()
  pair_count = 0
  stop_flag = 0
  new_sample_count = 0
  new_candidate_data = list(copy.deepcopy(candidate_data))

  for i in range(len(candidate_data)-1):
    sub_sequence_i, subseq_i_index = find_sub_seq(candidate_data[i].label, window_size, valid_tag_bar)
    for j in range(i+1, len(candidate_data)):
      sub_sequence_j, subseq_j_index = find_sub_seq(candidate_data[j].label, window_size, valid_tag_bar)
      same_subseq = extract_same_elem(sub_sequence_i, sub_sequence_j)
      # If the same subsequence exists:
      if same_subseq != []:
        for ii in range(len(sub_sequence_i)):
          for jj in range(len(sub_sequence_j)):
            if sub_sequence_i[ii] == sub_sequence_j[jj]:
              hyper_lambda = np.random.beta(hyper_alpha, hyper_alpha)
              newseq1, newseq2, newseq1_tag, newseq2_tag = lf_mixup(candidate_data, i, subseq_i_index[ii], j, subseq_j_index[jj], hyper_lambda)
              # add newseq1
              if score_limit_upper < 0:
                high_quality_1 = True
                high_quality_2 = True
              else:
                high_quality_1, score_1 = high_quality(newseq1, score_limit_upper, score_limit_low)
                high_quality_2, score_2 = high_quality(newseq2, score_limit_upper, score_limit_low)
              if high_quality_1:
                GUID_COUNT += 1
                new_candidate_data.append(InputExample(guid=GUID_COUNT, text_a=newseq1, text_b=None, label=newseq1_tag))
                new_sample_count += 1
              # add newseq2
              if high_quality_2:
                GUID_COUNT += 1
                new_candidate_data.append(InputExample(guid=GUID_COUNT, text_a=newseq2, text_b=None, label=newseq2_tag))
                new_sample_count += 1
              if high_quality_1 or high_quality_2:
                break
          break 
      if new_sample_count >= num_mixup:
        stop_flag = 1
        break
    if stop_flag:
      break
  time_end=time.time()
  print('{} extra samples are generated, the time cost is {} s'.format(new_sample_count, time_end - time_start))
  return new_candidate_data, new_sample_count


def tag2onehot(tag):
  label_map = {label : i for i, label in enumerate(label_list,1)}
  tagid = label_map[tag]
  onehot_tag = np.zeros(len(label_map)+1)
  onehot_tag[tagid] = 1
  return onehot_tag

def onehot2tag(tag):
  label_map = {label : i for i, label in enumerate(label_list,1)}
  reverse_label_map = {i:label for i, label in enumerate(label_list,1)}
  return_tag=[]
  for word in tag:
    if word.any()!=0:
      idx = np.where(word!=0)
      if len(idx[0]) > 1:
        mixtag = ''
        for i, item in enumerate(idx[0]):
          tag = reverse_label_map[item]
          mixtag += ' '+str(word[item])+ tag 
      else:
        mixtag = reverse_label_map[idx[0][0]]
      return_tag.append(mixtag)
  return return_tag


def slack_mixup(mixdata, sentence1_idx, sentence2_idx, start_idx1, start_idx2, hyper_lambda):
  '''
  This function implement sentence-level mixup, it will be called by the function augment().
  '''
  new_seq1 = list((mixdata[sentence1_idx].text_a).split())
  new_seq2 = list((mixdata[sentence2_idx].text_a).split())
  labels_1 = copy.deepcopy(mixdata[sentence1_idx].label)
  labels_2 = copy.deepcopy(mixdata[sentence2_idx].label)
  labels_1 = np.concatenate((['[CLS]'], labels_1, ['[SEP]']))
  labels_2 = np.concatenate((['[CLS]'], labels_2, ['[SEP]']))

  # Transfer to one-hot form
  new_seq1_tag = []
  new_seq2_tag = [] 
  for i, item in enumerate(labels_1):
    new_seq1_tag.append(tag2onehot(item))
  for i, item in enumerate(labels_2):
    new_seq2_tag.append(tag2onehot(item))                   
  # padding
  while len(new_seq1_tag) < args.max_seq_length:
    new_seq1_tag.append(np.zeros(12))
  while len(new_seq2_tag) < args.max_seq_length:
    new_seq2_tag.append(np.zeros(12))

  mix_seq = []
  mix_seq_tag = []

  # mixup
  for i in range(mix_len):
    e1 = wordidx2ebd[tokenizer.convert_tokens_to_ids(new_seq1[start_idx1[0]+i])]
    e2 = wordidx2ebd[tokenizer.convert_tokens_to_ids(new_seq2[start_idx2[0]+i])]
    e_mix = hyper_lambda*e1 + (1-hyper_lambda)*e2
    mix_token = most_similar(e_mix, 7) # return 1 candidate word
    exclude_pool = [new_seq1[start_idx1[0]+i], new_seq2[start_idx2[0]+i], '[UNK]', '[CLS]', '[SEP]', '[PAD]']

    for token in mix_token:
      #if token not in exclude_pool and token.find('[unused')==-1 and token.find('##')==-1:
      if token not in exclude_pool:
        mix_seq.append(token)
        break
    tag1 = new_seq1_tag[start_idx1[0]+i]
    tag2 = new_seq2_tag[start_idx2[0]+i]

    mix_tag = hyper_lambda*tag1 + (1-hyper_lambda)*tag2
    mix_seq_tag.append(mix_tag)

  # substitution
  for i in range(mix_len):
    new_seq1[start_idx1[0]+i] = mix_seq[i]
    new_seq2[start_idx2[0]+i] = mix_seq[i]
    new_seq1_tag[start_idx1[0]+i] = mix_seq_tag[i]
    new_seq2_tag[start_idx2[0]+i] = mix_seq_tag[i]

  new_seq1 = ' '.join(new_seq1)
  new_seq2 = ' '.join(new_seq2)
  return new_seq1, new_seq2, new_seq1_tag, new_seq2_tag


def slack_augment(candidate_data=None, num_mixup=None, hyper_alpha=8, score_limit_upper=500, score_limit_low=0):
  '''
  Given the candidate dataset and number of samples to be generated via mixup method, augment the training dataset by implementing mixup.
  Implement augmentation via slack-mixup
  '''
  global GUID_COUNT
  time_start=time.time()
  new_sample_count = 0
  stop_flag = 0
  mixup_data = []
  mixup_label = []
  for i in range(len(candidate_data)-1):
    sub_sequence_i, subseq_i_index = find_sub_seq(candidate_data[i].label, window_size, valid_tag_bar)
    if len(sub_sequence_i)>0:
      for j in range(i+1, len(candidate_data)):
        sub_sequence_j, subseq_j_index = find_sub_seq(candidate_data[j].label, window_size, valid_tag_bar)
        # If the slack pair exists:
        if len(sub_sequence_j)>0:
          hyper_lambda = np.random.beta(hyper_alpha, hyper_alpha) # Beta distribution
          newseq1, newseq2, newseq1_tag, newseq2_tag = slack_mixup(candidate_data, i, j, subseq_i_index, subseq_j_index, hyper_lambda)
          if score_limit_upper < 0:
            high_quality_1 = True
            high_quality_2 = True
          else:
            high_quality_1,score_1 = high_quality(newseq1, score_limit_upper, score_limit_low)
            high_quality_2,score_2 = high_quality(newseq2, score_limit_upper, score_limit_low)

          if high_quality_1 or high_quality_2:
            GUID_COUNT += 1
            mixup_data.append(InputExample(guid=GUID_COUNT, text_a=newseq1, text_b=None, label=candidate_data[i].label))
            mixup_label.append(newseq1_tag)
            new_sample_count += 1
          if new_sample_count >= num_mixup:
            stop_flag = 1
            break
          # add newseq2
          if high_quality_2:
            GUID_COUNT += 1
            mixup_data.append(InputExample(guid=GUID_COUNT, text_a=newseq2, text_b=None, label=candidate_data[j].label))
            mixup_label.append(newseq2_tag)
            new_sample_count += 1
          if new_sample_count >= num_mixup:
            stop_flag = 1
            break
    if stop_flag:
      break
  time_end=time.time()
  print('{} extra samples are generated, the time cost is {} s'.format(new_sample_count, time_end - time_start))
  return mixup_data, mixup_label, new_sample_count

def soft_mixup(candidate_1, candidate_2, hyper_lambda):
  '''
  This function implement sentence-level mixup, it will be called by the function soft_augment().
  '''
  # sparse sequence and label
  seq1 = list((candidate_1.text_a).split())
  seq2 = list((candidate_2.text_a).split())

  y1 = copy.deepcopy(candidate_1.label)
  y2 = copy.deepcopy(candidate_2.label)

  # Transfer to one-hot form
  new_seq1_tag = []
  new_seq2_tag = [] 
  for i, item in enumerate(y1):
    new_seq1_tag.append(tag2onehot(item))
  for i, item in enumerate(y2):
    new_seq2_tag.append(tag2onehot(item))                   
  # padding
  while len(new_seq1_tag) < args.max_seq_length:
    new_seq1_tag.append(np.zeros(12))
  while len(new_seq2_tag) < args.max_seq_length:
    new_seq2_tag.append(np.zeros(12))

  # prepare the generation form
  new_seq = copy.deepcopy(seq1) 
  new_seq_tag = copy.deepcopy(new_seq1_tag)

  assert len(seq1) == len(seq2), 'The two sequences should be in same valid length'
  mix_len_sentence = len(seq1)
  mix_seq = []
  mix_seq_tag = []

  # mixup
  for i in range(mix_len_sentence):
    e1 = wordidx2ebd[tokenizer.convert_tokens_to_ids(seq1[i])]
    e2 = wordidx2ebd[tokenizer.convert_tokens_to_ids(seq2[i])]
    e_mix = hyper_lambda*e1 + (1-hyper_lambda)*e2
    mix_token = most_similar(e_mix, 7)
    exclude_pool = [seq1[i], seq2[i], '[UNK]', '[CLS]', '[SEP]', '[PAD]']

    for token in mix_token:
      if token not in exclude_pool:
        mix_seq.append(token)
        break
    tag1 = new_seq1_tag[i]
    tag2 = new_seq2_tag[i]
    mix_tag = hyper_lambda*tag1 + (1-hyper_lambda)*tag2
    mix_seq_tag.append(mix_tag)

  # substitution
  for i in range(mix_len_sentence):
    new_seq[i] = mix_seq[i]
    new_seq_tag[i] = mix_seq_tag[i]
  new_seq = ' '.join(new_seq)
  return new_seq, new_seq_tag

def soft_augment(candidate_data=None, num_mixup=None, hyper_alpha=8, score_limit_upper=500, score_limit_low=0):
  global GUID_COUNT
  print('Implementing soft mixup augmentation, which may take hundreds of seconds')
  time_start=time.time()
  new_sample_count = 0
  mixup_data = []
  mixup_label = []
  candidate_data = copy.deepcopy(candidate_data)

  equal_len_index_list = soft_pair(candidate_data)
  pair_index_list = soft_pair_index_generator(equal_len_index_list, 15*num_mixup, valid_tag_bar)
  for index in pair_index_list:
    hyper_lambda = np.random.beta(hyper_alpha, hyper_alpha) # Beta distribution
    i = index[0]
    j = index[1]
    new_seq, new_seq_tag = soft_mixup(candidate_data[i], candidate_data[j], hyper_lambda)
    # add to the training set
    if score_limit_upper < 0:
      high_quality_flag = True
    else:
      high_quality_flag, score = high_quality(new_seq, score_limit_upper, score_limit_low)
    if high_quality_flag:
      GUID_COUNT += 1
      mixup_data.append(InputExample(guid=GUID_COUNT, text_a=new_seq, text_b=None, label=candidate_data[i].label))
      mixup_label.append(new_seq_tag)
      new_sample_count += 1
      case_util(score, candidate_data[i].text_a, candidate_data[j].text_a, candidate_data[i].label, candidate_data[j].label,
                new_seq, new_seq_tag, prefix='Soft_case')
    if new_sample_count >= num_mixup:
      break

  time_end=time.time()
  print('{} extra samples are generated, the time cost is {} s'.format(new_sample_count, time_end - time_start))
  return mixup_data, mixup_label, new_sample_count

def active_augment_learn(init_flag=None, train_data=None, num_initial=200, 
              active_policy=uncertainty_sampling, augment_method=lf_augment,
              num_query=5, num_sample=[100, 100, 100, 100, 100],  
              augment_rate=0.2, augment_decay=1, 
              hyper_alpha=8, alpha_decay=1, 
              Epochs=10, score_limit_low=0, score_limit_upper=500, fit_only_new_data=False, 
              mixup_flag=True, single_use=False, prefix='SeqMix'):
  '''
  Implement active learning initializaiton and learning loop
  '''
  func_paras = locals()
  # Data Initialization
  pool = copy.deepcopy(train_data)
  train_data = copy.deepcopy(train_data)
  original_datasize = len(train_data)

  initial_idx = np.random.choice(range(len(train_data)), size=num_initial, replace=False)
  train_data = np.array(train_data)[initial_idx]

  init_data_loader, query_idx = get_tr_set(size=num_initial, train_examples=train_data)
  pool = np.delete(pool, query_idx, axis=0)
  print(np.array(pool).shape)
  if init_flag:
    init_dir = 'init_dir'
    model = Ner.from_pretrained(init_dir)
    print("Initial model loaded from google drive")
  else:
    model = active_train(init_data_loader, None, Epochs)

  # report
  report = evaluate('Intialization', model)
  print_table = PrettyTable(['Model', 'Number of Query', 'Data Usage', 'Data Augmented', 'Test_F1'])
  print_table.add_row(['Initial Model', 'Model Initialization', len(train_data)/original_datasize, 0, report.split()[-2]])
  print(print_table)

  # augment on the seed set
  test_f1 = []
  dev_f1 = []
  num_augment = int(num_initial*augment_rate)

  if augment_method == slack_augment or soft_augment:
      soft_data, soft_labels, new_sample_count = augment_method(train_data, num_augment, hyper_alpha, score_limit_upper, score_limit_low)
      soft_loader = get_tr_set(train_examples=soft_data, soft_labels=soft_labels)
  else:
      mix_data, new_sample_count = augment_method(train_data, num_augment, hyper_alpha, score_limit_upper, score_limit_low)
      soft_loader = None

  aug_data_loader = get_tr_set(train_examples=train_data)
  model = active_train(data_loader=aug_data_loader, model=model, Epochs=Epochs, soft_loader=soft_loader)
  #return model
  report = evaluate('SeedSetAug', model)
  aug_total_count = new_sample_count
  print_table.add_row(['Augment Model', 'Seed Set Augmented', len(train_data)/original_datasize, aug_total_count, report.split()[-2]])
  print(print_table)
  save_result(prefix=prefix, func_paras=func_paras, report=report, table=print_table)

  # learning loop
  print('Learning loop start')
  for idx in range(num_query):
    num_augment = int((num_sample[idx]*augment_rate) *(augment_decay**idx))
    hyper_alpha = hyper_alpha * (alpha_decay**idx)

    print('Query no. %d' % (idx + 1))
    query_idx, query_instance = active_policy(model, pool, num_sample[idx])
    mixup_candidate = pool[query_idx]
    pool = np.delete(pool, query_idx, axis=0)

    if augment_method == slack_augment or soft_augment:
      new_soft_data, new_soft_labels, new_sample_count = augment_method(mixup_candidate, num_augment, hyper_alpha, score_limit_upper, score_limit_low)
      soft_data = np.concatenate((soft_data, new_soft_data))
      soft_labels = np.concatenate((soft_labels, new_soft_labels))
      soft_loader = get_tr_set(train_examples=soft_data, soft_labels=soft_labels)
      mix_data = mixup_candidate
    else:
      if mixup_flag: # mixup augment
        # mix_data consist of original mixup_candidate and new samples generated by SeqMix
        mix_data, new_sample_count = augment_method(mixup_candidate, num_augment, hyper_alpha, score_limit_upper, score_limit_low)
        soft_loader = None
      else: # duplicate original paring data
        mix_data, new_sample_count = duplicate_pair_data(mixup_candidate_X, mixup_candidate_y, num_augment)
    
    train_data = np.concatenate((train_data, mix_data))
    aug_total_count += new_sample_count
    aug_data_loader = get_tr_set(train_examples=train_data)
    model = active_train(data_loader=aug_data_loader, model=model, Epochs=Epochs, soft_loader=soft_loader)
    if single_use:
      train_data = train_data[:-new_sample_count]
      aug_total_count = new_sample_count
    report = evaluate('SeqMixAug', model)

    data_usage = len(train_data)
    if augment_method == lf_mixup:
      data_usage -= aug_total_count
    print_table.add_row(['Augmented Model', idx+1, data_usage/original_datasize, aug_total_count, report.split()[-2]])
    print(print_table)
    save_result(prefix=prefix, func_paras=func_paras, report=report, table=print_table)

  return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", type=str, default='bert-base-cased')
    parser.add_argument("--data_dir", type=str, default='data/')
    parser.add_argument("--do_eval", type=bool, default=True)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--task_name", type=str, default='ner')
    parser.add_argument("--output_dir", type=str, default='CoNLL/result')
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--prefix", type=str, default='file_save_name')
    parser.add_argument("--active_policy", type=str, default='nte')
    parser.add_argument("--augment_method", type=str, default='soft')
    parser.add_argument("--augment_rate", type=float, default=0.2)
    parser.add_argument("--hyper_alpha", type=float, default=8)



    # keep as default
    parser.add_argument("--server_ip", type=str, default='')
    parser.add_argument("--server_port", type=str, default='')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_cuda", type=bool, default=False)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--do_lower_case", type=bool, default=False)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--learning_rate", type=float, default=5e-05)
    parser.add_argument("--fp16", type=bool, default=False)
    parser.add_argument("--fp16_opt_level", type=str, default='O1')
    parser.add_argument("--eval_on", type=str, default='dev')
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)


    args = parser.parse_args()

    sp_output_dir = 'out_conll/'
    wordidx2ebd, ebd2wordidx = get_word_embedding(sp_output_dir)
    mydict_values = np.array(list(wordidx2ebd.values()))
    mydict_keys = np.array(list(wordidx2ebd.keys()))

    '''Scoring'''
    # Load pre-trained model (weights)
    gpt_model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
    gpt_model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    gpt_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    window_size = 5
    valid_tag_bar = 3
    mix_len = 5
    GUID_COUNT = 14041

    logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    if args.server_ip and args.server_port:
      # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
      print("Waiting for debugger attach")
      ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
      ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
      device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
      #n_gpu = torch.cuda.device_count()
      n_gpu = 1
    else:
      torch.cuda.set_device(args.local_rank)
      device = torch.device("cuda", args.local_rank)
      n_gpu = 1
      # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
      torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
      raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
      raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

    processor = NerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0 
    if args.do_train:
      train_examples = processor.get_train_examples(args.data_dir)
      num_train_optimization_steps = int(len(train_examples)/args.train_batch_size/args.gradient_accumulation_steps)*args.num_train_epochs
      if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.do_eval:
      if args.eval_on == 'dev':
        dev_examples = processor.get_dev_examples(args.data_dir)
      if args.eval_on == 'test':
        dev_examples = processor.get_test_examples(args.data_dir)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # prepare model
    config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
    model = Ner.from_pretrained(args.bert_model, from_tf = False, config = config)

    if args.local_rank == 0:
      torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

    # For our experiment, the following can be ignored
    if args.fp16:
      try:
        from apex import amp
      except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
      model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
      model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    if args.active_policy=='random':
      active_policy = random_sampling
    if args.active_policy=='lc':
      active_policy = uncertainty_sampling
    if args.active_policy=='nte':
      active_policy = nte_sampling

    if args.augment_method=='lf':
      augment_method = lf_augment
    if args.augment_method=='slack':
      augment_method = slack_augment
    if args.augment_method=='soft':
      augment_method = soft_augment

    soft_model = active_augment_learn(init_flag=False, train_data=train_examples, augment_rate=args.augment_rate, hyper_alpha=args.hyper_alpha, active_policy=active_policy, augment_method=augment_method, prefix=args.prefix, Epochs=10)