import os
import sys
import glob
import time
import argparse
import pandas as pd
import re
from SPARQLWrapper import SPARQLWrapper, JSON
from http.client import RemoteDisconnected

# Existing functions like get_count_relation_from_sparql, fetch_triple_score, etc., remain unchanged

def format_triples(triples, topk): 
    formatted_triples = []
    for triple in triples[:topk]:
        triple_tokens = triple.split('\t')
        # head
        head = ' <H> ' + triple_tokens[0].split('/')[-1]
        # relation
        try: 
            clean_relation = triple_tokens[1].split('/')[-1]
        except Exception as e:
            print(triple_tokens)
            print(e)
            break 
        clean_relation = re.sub(r'.*#', '', clean_relation)        
        relation = ' <R> ' + clean_relation
        # tail
        if triple_tokens[2].startswith('http://dbpedia.org/'):  # check if the tail is not literal
            tail = ' <T> ' + triple_tokens[2].split('/')[-1]
        else:
            clean_literal = re.sub(r'\^\^<http.*', '', triple_tokens[2])
            clean_literal = clean_literal.replace('"', '')
            clean_literal = clean_literal.replace('@e', '')
            tail = ' <T> ' + clean_literal
        formatted_triples.append(head + relation + tail)
    return formatted_triples

def process_triples_from_ranking(file_path, topk):
    df = pd.read_csv(file_path, delimiter='\t', quotechar='"', header=None, index_col=None, engine="python")
    df = df.applymap(lambda x: x.replace("(", "").replace(")", "").replace("'", "") if x is not None else x)
    df['triples'] = df.apply(lambda x: '\t'.join(x.astype(str)), axis=1)
    triples = df['triples'].tolist()
    formatted_triples = format_triples(triples, topk)
    return formatted_triples

def save_triples(formatted_triples, fname):
    with open(fname, 'w') as output_file:
        for triple in formatted_triples:
            output_file.write(f"{triple}")
            
def construct_file_paths(base_model, kge_model, gpt_model, combined_model, dataset, semantic_constraints):
    # Determine the subdirectory based on whether semantic constraints are applied
    dir_ = "/semantic-constraints" if semantic_constraints else "/non-semantic-constraints"

    # Base paths for each type of model
    base_paths = {
        "KGE": f"../data/{dataset}/predictions/KGE{dir_}/{kge_model}/",
        "LLM": f"../data/{dataset}/predictions/LLM/{gpt_model}/",
        "ANTS": f"../data/{dataset}/predictions/ANTS{dir_}/{combined_model}/"
    }

    # Determine the correct base path based on the base model
    if base_model not in base_paths:
        raise ValueError("Invalid base model specified.")

    base_path = base_paths[base_model]
    
    # Determine the subdirectory for ranking or triples based on the base model
    ranking_or_triples_dir = "triples" if base_model != "ANTS" else "ranking"

    return base_path, ranking_or_triples_dir

def process_ranking(kge_model, gpt_model, dataset, combined_model, base_model, semantic_constraints=True, topk=20):
    base_path, ranking_or_triples_dir = construct_file_paths(base_model, kge_model, gpt_model, combined_model, dataset, semantic_constraints)
    files = glob.glob(os.path.join(base_path, ranking_or_triples_dir, "*"))
    triples_ranking_dir = os.path.join(base_path, "ranking")
    triples_formatted_dir = os.path.join(base_path, "triples-formatted")
    
    if not os.path.exists(triples_ranking_dir):
        os.makedirs(triples_ranking_dir)
    
    if not os.path.exists(triples_formatted_dir):
        os.makedirs(triples_formatted_dir)
    
    for num, file in enumerate(files):
        filename = os.path.basename(file)
        entity_name = filename.replace(".txt", "").strip()
        print(f"{entity_name}: Generating triples-ranking {num+1}/{len(files)}")
        
        if os.path.isfile(f"{triples_ranking_dir}/{filename}"):
            print(f"File already exists: {triples_ranking_dir}/{filename}")
            continue
        
        with open(file, "r") as f:
            kge_triples = f.readlines()
        
        triples = format_triples_gpt(kge_triples[:5])  # This function is assumed to exist as per your previous code
        
        if base_model == "ANTS":
            with open(f"../data/{dataset}/predictions/LLM/{gpt_model}/ranking/{filename}") as f:
                gpt_triples = f.readlines()
            triples.extend(format_triples_gpt(gpt_triples))
        
        triple_scoring = {triple: fetch_triple_score(triple[1]) for triple in triples}
        sorted_triples = sorted(triple_scoring.items(), key=lambda item: item[1], reverse=True)
        
        with open(f"{triples_ranking_dir}/{filename}", "w") as f:
            for triple, score in sorted_triples:
                h, r, t = triple
                f.write(f"{h}\t{r}\t{t}\n")

    print("Converting triples into formatted triples for verbalizing inputs are started ...!")
    for num, file in enumerate(files):
        filename = os.path.basename(file)
        # Process the generated ranking file to create formatted triples
        formatted_triples = process_triples_from_ranking(f"{triples_ranking_dir}/{filename}", topk)
        save_triples(formatted_triples, f"{triples_formatted_dir}/{filename}")
    print("Converting triples into formatted triples for verbalizing inputs are done ...!")
def main():
    parser = argparse.ArgumentParser(description='Process triples ranking with specified models.')
    
    parser.add_argument('--kge_model', type=str, help='The name of the KGE model')
    parser.add_argument('--gpt_model', type=str, help='The name of the GPT model')
    parser.add_argument('--dataset', type=str, required=True, help='The name of the dataset')
    parser.add_argument('--combined_model', type=str, help='The combined model name')
    parser.add_argument('--base_model', type=str, required=True, help='The base model name')
    parser.add_argument('--semantic_constraints', action='store_true', help='Apply semantic constraints')
    parser.add_argument('--topk', type=int, default=20, help='Top K triples to format and save')
    
    args = parser.parse_args()

    process_ranking(
        kge_model=args.kge_model,
        gpt_model=args.gpt_model,
        dataset=args.dataset,
        combined_model=args.combined_model,
        base_model=args.base_model,
        semantic_constraints=args.semantic_constraints,
        topk=args.topk
    )

if __name__ == '__main__':
    main()
