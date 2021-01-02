import argparse
import csv
import json
import logging
import os
import random
import sys
import copy
import math
import time
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


def get_tr_set(size=None, train_examples=None, batch_size=32, soft_labels=[], args=None):
  train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer, logger)
  if size: # return part of features
    select_idx = np.random.choice(range(len(train_features)), size=size, replace=False)
    train_features = list(np.array(train_features)[select_idx])

  logger.info("  Num examples = %d", len(train_examples))
  all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
  all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
  all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
  if len(soft_labels):
    all_label_ids = torch.tensor([soft_label for soft_label in soft_labels], dtype=torch.float64)
  else:
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
  train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids, all_lmask_ids)
  if args.local_rank == -1:
    train_sampler = RandomSampler(train_data)
  else:
    train_sampler = DistributedSampler(train_data)

  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
  if size:
    return train_dataloader, select_idx
  return train_dataloader

def get_eval_set(eval_on, eval_batch_size=8):
  if eval_on == "dev":
    eval_examples = processor.get_dev_examples(args.data_dir)
  elif eval_on == "test":
    eval_examples = processor.get_test_examples(args.data_dir)
  else:
    raise ValueError("eval on dev or test set only")
  eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, logger)
  logger.info("***** Running evaluation *****")
  logger.info("  Num examples = %d", len(eval_examples))
  logger.info("  Batch size = %d", args.eval_batch_size)
  all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
  all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
  all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
  all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
  all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
  all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
  eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_valid_ids,all_lmask_ids)
  # Run prediction for full data
  eval_sampler = SequentialSampler(eval_data)
  eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
  return eval_dataloader

'''Evaluation'''
def evaluate(prefix=None, model=None, args=None):
    eval_dataloader = get_eval_set(eval_on=args.eval_on, eval_batch_size=args.eval_batch_size)
    model.to(device)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    raw_logits = []
    label_map = {i : label for i, label in enumerate(label_list,1)}
    for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
        
        #raw_logits.append(logits)
        logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j,m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_map):
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    break
                else:
                    temp_1.append(label_map[label_ids[i][j]])
                    try:
                      temp_2.append(label_map[logits[i][j]])
                    except:
                      temp_2.append('UKN')

    report = classification_report(y_true, y_pred, digits=4)
    logger.info("\n%s", report)
    return report

def save_result(prefix='Active', func_paras=None, report=None, table=None, output_dir=None):
  result_path = os.path.join(output_dir, prefix+'.txt')
  with open(result_path,'a') as f:
    if func_paras:
      for para in func_paras:
        if(type(func_paras[para]))==np.ndarray:
          func_paras[para] = func_paras[para].shape
        if(type(func_paras[para]))==list:
          func_paras[para] = np.array(func_paras[para]).shape 
      f.write('\nParameters:\n')
      for item in func_paras.items():
        f.write(str(item)+'\n')
    if report:
      f.write(report)
    if table:
      table = table.get_string()
      f.write(table)

def multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    """
    Selects the indices of the n_instances highest values.

    Input:
      values: Contains the values to be selected from.
      n_instances: Specifies how many indices to return.
    Output:
      The indices of the n_instances largest values.
    """
    assert n_instances <= values.shape[0], 'n_instances must be less or equal than the size of utility'

    max_idx = np.argpartition(-values, n_instances-1, axis=0)[:n_instances]
    return max_idx

def uncertainty_sampling(model_instance, pool, size):
  '''
  Uncertainty sampling policy.

  Input:
    model_instance: the model to do the uncertainty measure by give the labels prediction over unobserved data.
    pool: the unobserved data.
    size: the number of instances to be sampled in each round.
  Output:
    query_index: the n_instances index of sampled data.
    pool[query_index]: the corresponding data.
  '''
  active_eval_loader = get_tr_set(train_examples=pool, batch_size=1, args=args)
  raw_prediction, turncate_list = active_eval(active_eval_loader, model_instance) # predict, get the softmax output
  word_prob = np.max(raw_prediction,axis=2) # select the max probability prediction as the word tag
  sentence_uncertainty = []
  for i, sentence in enumerate(word_prob):
    sentence_uncertainty.append(np.sum(1-sentence[:turncate_list[i]]))
  query_index = multi_argmax(np.array(sentence_uncertainty), size)
  return query_index, pool[query_index]

def nte_sampling(model_instance, pool, size):
  active_eval_loader = get_tr_set(train_examples=pool, batch_size=1, args=args)
  raw_prediction, turncate_list = active_eval(active_eval_loader, model_instance) # predict, get the softmax output
  sentence_nte = cal_nte(raw_prediction, turncate_list)
  query_index = multi_argmax(np.array(sentence_nte), size)
  return query_index, pool[query_index]

def cal_nte(logits, turncate):
  sentence_nte = []
  for idx, sent in enumerate(logits):
    sent_sum = 0
    for word in sent[:turncate[idx]]:
      tag_sum = 0
      for tag in word:
        tag_sum += tag*math.log(tag)
      sent_sum += tag_sum
    sentence_nte.append(-sent_sum/turncate[idx])
  return sentence_nte

def qbc_sampling(model_com, pool, n_instance):
  com_pred = []
  active_eval_loader = get_tr_set(train_examples=pool, batch_size=1, args=args)
  for _model in model_com:
    raw_prediction, turncate_list = active_eval(active_eval_loader, _model)
    tag_prediction = result2tag(raw_prediction, turncate_list)
    com_pred.append(tag_prediction)
  vote_entropy = cal_vote_entropy(com_pred)
  query_index = multi_argmax(vote_entropy, n_instance)
  return query_index, pool[query_index]

def cal_vote_entropy(mc_pred):
  '''
  Calculate the vote entropy

  Input:
    mc_pred: 3d-shape (num_mc_model * num_sentence * max_len * n_tags)
  Output:
    vote_entropy: 2d-shape (num_sentence * max_len)
  '''
  num_mc_model = len(mc_pred)
  num_sentence = mc_pred[0].shape[0]

  print('vote_matrix')
  vote_matrix = np.zeros((num_sentence, args.max_seq_length, num_labels))
  for model_idx, pred in enumerate(mc_pred):
    for s_idx, sentence in enumerate(pred):
      for w_idx, word in enumerate(sentence):
        vote_matrix[s_idx][w_idx][word] += 1
  print('vote_prob_matrix')
  vote_prob_matrix = np.zeros((num_sentence, args.max_seq_length, num_labels))
  for s_idx, sentence in enumerate(vote_matrix):
    for w_idx, word in enumerate(sentence):
      for tag_idx in range(num_labels):
        prob_i = np.sum(word==tag_idx) / num_mc_model
        vote_prob_matrix[s_idx][w_idx][tag_idx] = prob_i
  print('vote_entropy')
  vote_entropy = np.zeros(num_sentence)
  for s_idx, sentence in enumerate(vote_prob_matrix):
    sentence_entropy = 0
    for w_idx, word in enumerate(sentence):
      word_entropy = 0
      for tag_prob in word:
        if tag_prob:
          word_entropy -= tag_prob*(math.log(tag_prob)) 
      sentence_entropy += word_entropy
    vote_entropy[s_idx] = sentence_entropy
  
  return vote_entropy

def result2tag(result, turncate):
  '''
  Convert the result with 3-d shape to the tags with 2-d shape. 
  '''
  sentences = []
  for idx, sentence in enumerate(result):
    valid_len = turncate[idx]
    words = []
    for word in sentence[:valid_len]:
      word = word.tolist()
      tag = word.index(max(word))
      words.append(tag)
    sentences.append(words)
  return np.array(sentences)

def random_sampling(model_instance, input_data, n_instances):
  '''
  Random sampling policy.

  Input:
    model_instance: model
    input_data: the unobserved data.
    n_instances: the number of instances to be sampled in each round.
  Output:
    query_index: the n_instances index of sampled data.
    input_Data[query_index]: the corresponding data.
  '''
  query_index = np.random.choice(range(len(input_data)), size=n_instances, replace=False)
  return query_index, input_data[query_index]

def active_train(data_loader=None, model=None, Epochs=5, soft_loader=None, args=None):
  config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
  if model==None:
    model = Ner.from_pretrained(args.bert_model, from_tf = False, config = config)
  return_model = Ner.from_pretrained(args.bert_model, from_tf = False, config = config)
  model.to(device)
  return_model.to(device)
  param_optimizer = list(model.named_parameters())
  no_decay = ['bias','LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]
  num_train_optimization_steps = int(len(data_loader.dataset)/args.train_batch_size/args.gradient_accumulation_steps)*args.num_train_epochs #2190
  warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
  
  current_train_size = 0
  if soft_loader:
    current_train_size = len(data_loader.dataset) + len(soft_loader.dataset)
  else:
    current_train_size = len(data_loader.dataset)
  print('Training on {} data'.format(current_train_size))

  model.train()
  tr_loss = 2020
  for epoch_idx in trange(int(Epochs), desc="Epoch"):
    current_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, valid_ids,l_mask = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)
        if n_gpu > 1:
            loss = loss.mean() # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        current_loss += loss
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
    if soft_loader:
      for input_ids, input_mask, segment_ids, soft_labels, valid_ids,l_mask in tqdm(soft_loader, desc="Soft Training"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            soft_labels = soft_labels.to(device)
            l_mask = l_mask.to(device)
            #with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
            #logits = F.softmax(logits, dim=2)
            logits = logits.detach().cpu().float()
            soft_labels = soft_labels.detach().cpu().float()
            pos_weight = torch.ones([num_labels])  # All weights are equal to 1
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = 0
            for i in range(len(logits)):
              turncate_len = np.count_nonzero(l_mask[i].detach().cpu().numpy())
              logit = logits[i][:turncate_len]
              soft_label = soft_labels[i][:turncate_len]
              loss += criterion(logit, soft_label)
            loss = Variable(loss, requires_grad=True)
            current_loss += loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
    if current_loss <= tr_loss:
      return_model.load_state_dict(model.state_dict())
      tr_loss = current_loss

  return return_model

def active_eval(active_data_loader=None, model=None):
    model.to(device)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    y_true = []
    y_pred = []
    raw_logits = []
    turncate_list = []
    label_map = {i : label for i, label in enumerate(label_list,1)}
    for input_ids, input_mask, segment_ids, label_ids,valid_ids,l_mask in tqdm(active_data_loader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        valid_ids = valid_ids.to(device)
        label_ids = label_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
          logits = model(input_ids, segment_ids, input_mask,valid_ids=valid_ids,attention_mask_label=l_mask)
        
        logits = F.softmax(logits, dim=2)
        assert logits.shape[0] == 1
        logits = logits.detach().cpu().numpy().reshape((logits.shape[1], logits.shape[2]))
        turncate_len = np.count_nonzero(l_mask.detach().cpu().numpy())
        turncate_list.append(turncate_len)
        raw_logits.append(logits)
    return raw_logits, turncate_list


def active_learn(init_flag=None, train_data=None, num_initial=200, 
                 active_policy=None, num_query=5, num_sample=[100, 100, 100, 100, 100], 
                 dev_data=None, fit_only_new_data=False, Epochs=10, prefix='Active', args=None):
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

  init_data_loader, query_idx = get_tr_set(size=num_initial, train_examples=train_data, args=args)
  pool = np.delete(pool, query_idx, axis=0)
  print(np.array(pool).shape)
  if init_flag:
    init_dir = 'init_dir'
    model = Ner.from_pretrained(init_dir)
  else:
    model = active_train(init_data_loader, None, Epochs, args=args)

  report = evaluate('Intialization', model, args)
  print_table = PrettyTable(['Model', 'Number of Query', 'Data Usage', 'Test_F1'])
  print_table.add_row(['Active Model', 'Model Initialization', len(train_data)/original_datasize, report.split()[-2]])
  print(print_table)
  save_result(prefix=args.prefix, report=report, table=print_table, output_dir=args.output_dir)

  print('Learning loop start')
  for idx in range(num_query):
      print('\n\n-------Query no. %d--------\n' % (idx + 1))
      query_idx, query_instance = active_policy(model, pool, num_sample[idx])

      if fit_only_new_data:
        train_data = pool[query_idx]
      else:
        train_data = np.concatenate((train_data, pool[query_idx]))
      pool = np.delete(pool, query_idx, axis=0)
      active_data_loader = get_tr_set(train_examples=train_data, args=args)
      model = active_train(active_data_loader, model, Epochs, args=args)

      report = evaluate('Active Learning', model, args)
      print_table.add_row(['Active Model', idx+1, len(train_data)/original_datasize, report.split()[-2]])
      print(print_table)

      save_result(prefix=args.prefix, func_paras=func_paras, report=report, table=print_table, output_dir=args.output_dir)

  return model

'''
def active_qbc_learn(init_flag=None, train_data=train_examples, num_initial=200, 
                 active_policy=qbc_sampling, num_com=3, num_query=5, num_sample=[100, 100, 100, 100, 100], 
                 dev_data=dev_examples, fit_only_new_data=False, Epochs=10, prefix='Active'):
  #Implement active learning initializaiton and learning loop
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

  report = evaluate('Intialization', model)
  print_table = PrettyTable(['Model', 'Number of Query', 'Data Usage', 'Test_F1'])
  print_table.add_row(['Active Model', 'Model Initialization', len(train_data)/original_datasize, report.split()[-2]])
  print(print_table)

  # Construct the committee
  model_com = []
  config = BertConfig.from_pretrained(args.bert_model, num_labels=num_labels, finetuning_task=args.task_name)
  for i in range(num_com):
    _model = Ner.from_pretrained(args.bert_model, from_tf = False, config = config)
    _model.load_state_dict(model.state_dict())
    model_com.append(_model)


  print('Learning loop start')
  for idx in range(num_query):
      print('\n-------Query no. %d--------\n' % (idx + 1))
      query_idx, query_instance = active_policy(model_com, pool, num_sample[idx])

      if fit_only_new_data:
        train_data = pool[query_idx]
      else:
        train_data = np.concatenate((train_data, pool[query_idx]))
      pool = np.delete(pool, query_idx, axis=0)
      active_data_loader = get_tr_set(train_examples=train_data)
      for _idx, _model in enumerate(model_com):
        print('\n-------Committee no. %d--------\n' % (_idx + 1))
        _model = active_train(active_data_loader, _model, Epochs)
        report = evaluate('Active Learning', _model)
        print_table.add_row(['Active Model', idx+1, len(train_data)/original_datasize, report.split()[-2]])
        print(print_table)

      save_result(prefix=prefix, func_paras=func_paras, report=report, table=print_table)

  return model
'''

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
        # parse args
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

    model = active_learn(init_flag=False, train_data=train_examples, dev_data=dev_examples, active_policy=active_policy, prefix=args.prefix, Epochs=args.num_train_epochs, args=args)
