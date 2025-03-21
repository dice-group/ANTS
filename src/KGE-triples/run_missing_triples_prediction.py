import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import math
import glob

from os.path import join
import torch.backends.cudnn as cudnn
import pandas as pd
from tqdm import tqdm
from typing import Tuple

from evaluation import ranking_and_hits
from model import DistMultLiteral, ComplexLiteral, ConvELiteral, DistMultLiteral_gate,ComplexLiteral_gate, ConvELiteral_gate, DistMultLiteral_gate_text, ComplexLiteral_gate_text, ConvELiteral_gate_text

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer
from spodernet.preprocessing.processors import ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper, ApplyFunction, StreamToBatch
from spodernet.utils.global_config import Config, Backends
from spodernet.utils.logger import Logger, LogLevel
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
from spodernet.hooks import LossHook, ETAHook
from spodernet.utils.util import Timer
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.utils.cuda_utils import CUDATimer
from spodernet.preprocessing.processors import TargetIdx2MultiTarget
np.set_printoptions(precision=3)
import pdb
timer = CUDATimer()
cudnn.benchmark = True

# parse console parameters and set global variables
Config.backend = Backends.TORCH
Config.parse_argv(sys.argv)

Config.cuda = True
Config.embedding_dim = 200
#Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG

# Random seed
from datetime import datetime
rseed = int(datetime.now().timestamp())
print(f'Random seed: {rseed}')
np.random.seed(rseed)
torch.manual_seed(rseed)
torch.cuda.manual_seed(rseed)

#model_name = 'DistMult_{0}_{1}'.format(Config.input_dropout, Config.dropout)
model_name = '{2}_{0}_{1}_literal'.format(Config.input_dropout, Config.dropout, Config.model_name)
epochs = Config.epochs
load = True
if Config.dataset is None:
    Config.dataset = 'FB15k-237'
model_path = 'saved_models/{0}_{1}.model'.format(Config.dataset, model_name)


''' Preprocess knowledge graph using spodernet. '''
def preprocess(dataset_name, delete_data=False):
    full_path = 'data/{0}/e1rel_to_e2_full.json'.format(dataset_name)
    train_path = 'data/{0}/e1rel_to_e2_train.json'.format(dataset_name)
    dev_ranking_path = 'data/{0}/e1rel_to_e2_ranking_dev.json'.format(dataset_name)
    test_ranking_path = 'data/{0}/e1rel_to_e2_ranking_test.json'.format(dataset_name)

    keys2keys = {}
    keys2keys['e1'] = 'e1' # entities
    keys2keys['rel'] = 'rel' # relations
    #keys2keys['rel_eval'] = 'rel' # relations
    keys2keys['e2'] = 'e1' # entities
    keys2keys['e2_multi1'] = 'e1' # entity
    keys2keys['e2_multi2'] = 'e1' # entity
    input_keys = ['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2']
    d = DatasetStreamer(input_keys)
    d.add_stream_processor(JsonLoaderProcessors())
    d.add_stream_processor(DictKey2ListMapper(input_keys))
    # process full vocabulary and save it to disk
    d.set_path(full_path)
    p = Pipeline(Config.dataset, delete_data, keys=input_keys, skip_transformation=True)
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
    p.add_token_processor(AddToVocab())
    p.execute(d)
    p.save_vocabs()

    # process train, dev and test sets and save them to hdf5
    p.skip_transformation = False
    for path, name in zip([train_path, dev_ranking_path, test_ranking_path], ['train', 'dev_ranking', 'test_ranking']):
        d.set_path(path)
        p.clear_processors()
        p.add_sent_processor(ToLower())
        p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')),keys=['e2_multi1', 'e2_multi2'])
        p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), keys=['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2'])
        p.add_post_processor(StreamToHDF5(name, samples_per_file=1500 if Config.dataset == 'YAGO3-10' else 1000, keys=input_keys))
        p.execute(d)

def create_constraints(triples: np.ndarray) -> Tuple[dict, dict, dict, dict]:
    """
    (1) Extract domains and ranges of relations
    (2) Store a mapping from relations to entities that are outside of the domain and range.
    Create constraints entities based on the range of relations
    :param triples:
    :return:
    """
    assert isinstance(triples, np.ndarray)
    assert triples.shape[1] == 3

    # (1) Compute the range and domain of each relation
    domain_per_rel = dict()
    range_per_rel = dict()

    range_constraints_per_rel = dict()
    domain_constraints_per_rel = dict()
    set_of_entities = set()
    set_of_relations = set()
    print(f'Constructing domain and range information by iterating over {len(triples)} triples...', end='\t')
    for (e1, p, e2) in triples:
        # e1, p, e2 have numpy.int16 or else types.
        domain_per_rel.setdefault(p, set()).add(e1)
        range_per_rel.setdefault(p, set()).add(e2)
        set_of_entities.add(e1)
        set_of_relations.add(p)
        set_of_entities.add(e2)
    print(f'Creating constraints based on {len(set_of_relations)} relations and {len(set_of_entities)} entities...',
          end='\t')
    for rel in set_of_relations:
        range_constraints_per_rel[rel] = list(set_of_entities - range_per_rel[rel])
        domain_constraints_per_rel[rel] = list(set_of_entities - domain_per_rel[rel])
    return domain_constraints_per_rel, range_constraints_per_rel, domain_per_rel, range_per_rel

def main():
    confidence_score = 0.25
    if Config.process: preprocess(Config.dataset, delete_data=False)
    input_keys = ['e1', 'rel', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(Config.dataset, keys=input_keys)
    p.load_vocabs()
    vocab = p.state['vocab']

    num_entities = vocab['e1'].num_token

    numerical_literals = np.load(f'data/{Config.dataset}/literals/numerical_literals.npy', allow_pickle=True)
    text_literals = np.load(f'data/{Config.dataset}/literals/text_literals.npy', allow_pickle=True)

    # Normalize numerical literals
    max_lit, min_lit = np.max(numerical_literals, axis=0), np.min(numerical_literals, axis=0)
    numerical_literals = (numerical_literals - min_lit) / (max_lit - min_lit + 1e-8)

    # Load literal models
    if Config.model_name is None or Config.model_name == 'DistMult':
        model = DistMultLiteral_gate(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
    elif Config.model_name == 'ComplEx':
        model = ComplexLiteral_gate(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
    elif Config.model_name == 'ConvE':
        model = ConvELiteral_gate(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
    elif Config.model_name == 'DistMult_text':
        model = DistMultLiteral_gate_text(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals, text_literals)
    elif Config.model_name == 'ComplEx_text':
        model = ComplexLiteral_gate_text(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals, text_literals)
    elif Config.model_name == 'ConvE_text':
        model = ConvELiteral_gate_text(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals, text_literals)
    elif Config.model_name == 'DistMult_glin':
        model = DistMultLiteral(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
    elif Config.model_name == 'ComplEx_glin':
        model = ComplexLiteral(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
    elif Config.model_name == 'ConvE_glin':
        model = ConvELiteral(vocab['e1'].num_token, vocab['rel'].num_token, numerical_literals)
    else:
        raise Exception("Unknown model!")

    if Config.cuda:
        model.cuda()

    if load:
        model_params = torch.load(model_path)
        total_param_size = []
        params = [(key, value.size(), value.numel()) for key, value in model_params.items()]
        for key, size, count in params:
            total_param_size.append(count)
        model.load_state_dict(model_params)
        model.eval()
        print("entities number", vocab["e1"].num_token)
        print("relations number", vocab["rel"].num_token)
        ranks = []
        rel_list = []
        for rel in vocab["rel"].token2idx:
            rel_list.append(rel)
        e1_list = []
        for item in vocab["e1"].idx2token:
            e1_list.append(False)
        
        ds_name = "ESBM-DBpedia"
        kge_model = "complex"
        if not os.path.exists(f"../../data/{ds_name}/relevant-triples/KGE/{kge_model}"):
            os.makedirs(f"../../data/{ds_name}/relevant-triples/KGE/{kge_model}")
        if not os.path.exists(f"../../data/{ds_name}/relevant-triples/KGE/{kge_model}/triples"):
            os.makedirs(f"../../data/{ds_name}/relevant-triples/KGE/{kge_model}/triples")
        fr = open(f"../../data/{ds_name}/elist.txt", "r")
        content = fr.readlines()
        fr.close()
        entities = []
        for num, item in enumerate(content):
            if num == 0:
                continue
            item = item.replace("\n", "")
            item = item.split("\t")
            entity = item[1]
            entities.append(entity)
            
        for entity in tqdm(entities):
            #print(entity)
            entity = entity.split("/")[-1].strip()
            entity_name = entity
            entity =f"http://dbpedia.org/resource/{entity}"
            
            entity = entity.lower()
            #try:
            check_token = [vocab["e1"].token2idx[f"{entity}"]]
            #except:
            #continue
            triples = dict()
            for rel in rel_list:
                rel = rel.lower()
                if rel=="oov" or rel=="":
                    continue
                try:
                    check_token = [vocab["rel"].token2idx[f"{rel}"]]
                except:
                    continue
                elist = e1_list.copy()
                head = entity
                relation = rel

                rel_id = vocab["rel"].token2idx[rel]
                e1_tensor = torch.tensor([vocab["e1"].token2idx[f"{entity}"]]).cuda()
                rel_tensor = torch.tensor([vocab["rel"].token2idx[rel]]).cuda()
                scores = model.forward(e1_tensor, rel_tensor)
                pred_output = scores.view(1, -1).cpu()
                (output_top_scores, output_top) = torch.topk(pred_output, 5)
                scores = output_top_scores.squeeze(0).detach().numpy().tolist()
                pred_topk = output_top.squeeze(0).detach().numpy().tolist()
                predicted_tail = vocab["e1"].idx2token[pred_topk[0]]
                for num, score in enumerate(scores):
                    if score > confidence_score:
                        predicted_tail = vocab["e1"].idx2token[pred_topk[num]]
                        triple = (head, relation, predicted_tail)
                        triples[triple]=score
            
            if len(triples)>0:
                triples_sorted = sorted(triples.items(), key=lambda x:x[1], reverse=True)
                topk=len(triples_sorted)
                if len(triples_sorted)>20:
                    topk=20
                fw = open(f"../../../data/{ds_name}/relevant-triples/KGE/{kge_model}/triples/{entity_name}.txt", "w")
                triples = []
                triples_dict = {}
                for triple_score in triples_sorted[:topk]:
                    triple, score = triple_score
                    h, r, t = triple
                    relation = relation.split("/")[-1]
                    if (h, relation, t) not in triples_dict:
                        triples_dict[h, relation, t] = len(triples_dict)
                        triples.append((h, r, t))
                triples = list(set(triples))
                for triple in triples:
                    h, r, t = triple
                    fw.write(f"{h}\t{r}\t{t}\n")
                fw.close()
            else:
                print(f"{entity} --> {len(triples)}")
if __name__ == '__main__':
    main()
