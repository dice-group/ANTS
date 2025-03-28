import argparse
import glob
import pandas as pd
import re
import os 
from pathlib import Path

def format_triples(triples, topk): 
    formatted_triples=[]
    for triple in triples[:topk]:
        triple = triple.replace("\n","")
        triple_tokens= triple.split('\t')
        head = ' <H> '+triple_tokens[0].split('/')[-1]
        try: 
            clean_relation= triple_tokens[1].split('/')[-1]
        except Exception as e:
            print(triple_tokens, e)
            break 
        clean_relation= re.sub(r'.*#', '', clean_relation)        
        relation = ' <R> '+clean_relation
        if triple_tokens[2].startswith('http://dbpedia.org/'): 
            tail= ' <T> '+triple_tokens[2].split('/')[-1]
        else:
            clean_literal= re.sub(r'\^\^<http.*', '', triple_tokens[2]).replace("'", "").replace('"', "").replace('@e','')
            tail= ' <T> '+clean_literal
        formatted_triples.append(head+relation+tail)
    return formatted_triples

def process_triples(present_, topk):
    entity_df = pd.read_csv(present_,sep=' .\n', header=None, index_col=None, engine='python')
    entity_df.columns=['triples']
    triples=entity_df['triples'].tolist()
    return format_triples(triples, topk)

def format_selected_triples_llm(triples, topk): 
    selected_triples=[]
    for triple in triples:
        if ";" in triple:
            data = triple.split(";")
        else:
            data = triple.split(",")
        if len(data)<3:
            continue
        h, r, t = data[0], data[1], " ".join(data[2:]) if len(data)>3 else data[2]
        if len(selected_triples) >=topk:
            continue
        if t.startswith('http://dbpedia.org/'):
            tail= t.split('/')[-1]
        else:
            clean_literal= re.sub(r'\^\^<http.*', '', t).replace("'", "").replace('"', "").replace('@e','')
            tail= clean_literal
            if tail[-1]==")":
                tail = tail[0:-1]
        if h[0]=="(":
            h=h[1:]
        head = "http://dbpedia.org/resource/"+h.strip().replace(" ", "_")
        relation = "http://dbpedia.org/ontology/"+r.strip()
        if "?" in tail or "wiki" in relation:
            continue
        selected_triples.append(f"{head}\t{relation}\t{tail}\n")
    return selected_triples

def save_triples(formatted_triples, fname):
    with open(fname, 'w') as output_file:
        for triple in formatted_triples:
            output_file.write(f"{triple}")

def main(args):
    if args.dataset=="ESSUM-DBpedia":
        dataset = "ESBM-DBpedia"
    elif args.dataset=="ESSUM-FACES":
        dataset = "FACES"
    else:
        raise NotImplementedError
    root_path_verbalization = f"../../data/{dataset}/predictions/LLM"
    system_dir = f"{root_path_verbalization}/{args.system}"
    triples_formatted_dir = f'{system_dir}/triples-formatted/'
    triples_selected_dir = f'{system_dir}/triples-selected/'
    os.makedirs(system_dir, exist_ok=True)
    os.makedirs(triples_selected_dir, exist_ok=True)    
    os.makedirs(triples_formatted_dir, exist_ok=True)
    
    triples_all_dir= f'../../data/{dataset}/predictions/LLM/{args.system}/triples'
    triples_all_list = glob.glob(triples_all_dir + "/*") 
    
    for ename in triples_all_list:
        print(ename)
        with open(ename) as f:
            content = [line.strip().replace('("', "").replace('")', "").replace('"', "") for line in f.readlines() if line.strip()]
        selected_triples = format_selected_triples_llm(content, args.topk)
        formatted_triples = format_triples(selected_triples, args.topk)
        fname = os.path.basename(ename)
        print("**********", len(selected_triples))
        save_triples(selected_triples, f"{triples_selected_dir}/{fname}")
        save_triples(formatted_triples, f"{triples_formatted_dir}/{fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and format triples")
    parser.add_argument("--system", type=str, required=True, help="System name (e.g., gpt-3.5)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., FACES)")
    parser.add_argument("--topk", type=int, default=20, help="Number of top triples to select")
    args = parser.parse_args()
    main(args)
