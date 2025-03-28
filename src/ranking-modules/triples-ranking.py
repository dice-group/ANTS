import argparse
import os
import glob
import time
from http.client import RemoteDisconnected
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import re

def get_count_relation_from_sparql(uri):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT (COUNT(DISTINCT CONCAT(str(?s), str(?o))) AS ?cardinality)
    WHERE {{
      ?s <{uri}> ?o .
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    cardinality = [result['cardinality']['value'].split('/')[-1] for result in results["results"]["bindings"]]
    return int(cardinality[0]) if cardinality else 0  # Return ["Unknown"] if no type was found

def format_triples_gpt(triples_text, topk=20): 
    formatted_triples=[]
    triples = []
    for triple in triples_text:
        if triple.strip()=="":
            continue
        if len(formatted_triples) >=topk:
            continue
        triple = triple.replace("\n","")
        triple = triple.replace("(","")
        triple = triple.replace(")","")
        triple = triple.replace("<","")
        triple = triple.replace(">","")
        triple = triple.replace("\"","")
        triple = triple.strip()
        triple_tokens= triple.split(',')
        if len(triple_tokens)>3:
            h = triple_tokens[0]
            r = triple_tokens[1].replace("\"","")
            r = triple_tokens[1].replace(" ","")
            t = ""
            for n, word in enumerate(triple_tokens[2:len(triple_tokens)]):
                if n==0:
                    t += f"{word}"
                else:
                    t += f", {word}"
        elif len(triple_tokens)==3:
            h = triple_tokens[0]
            r = triple_tokens[1].replace("\"","")
            r = triple_tokens[1].replace(" ","")
            t = triple_tokens[2]
        else:
           continue
        if "?" in t: # we remove uncomple
            continue
        if "wiki" in r: # we remove from wikidata information in this experiment
            continue
        if "abstract" in r: # we remove from abstract property information in this experiment
            continue
        triples.append((h.strip(), f"http://dbpedia.org/ontology/{r.strip()}", t.strip()))
    return triples

def fetch_triple_score(relation):
    try:
        return get_count_relation_from_sparql(relation)
    except RemoteDisconnected:
        time.sleep(5)
        return fetch_triple_score(relation)

def format_triples(triples, topk): 
    formatted_triples=[]
    for triple in triples[:topk]:
        triple_tokens= triple.split('\t')
        print("triple tokens", triple_tokens)
        # head
        head = ' <H> '+triple_tokens[0].split('/')[-1]
        # relation
        try: 
            clean_relation= triple_tokens[1].split('/')[-1]
        except Exception as e:
            print (triple_tokens)
            print (e)
            break 
        clean_relation= re.sub(r'.*#', '', clean_relation)        
        relation = ' <R> '+clean_relation
        #tail
        if triple_tokens[2].startswith('http://dbpedia.org/'): # check if the tail is not literal
            tail= ' <T> '+triple_tokens[2].split('/')[-1]
        else:
            clean_literal= re.sub(r'\^\^<http.*', '', triple_tokens[2])
            clean_literal=clean_literal.replace('"','')
            clean_literal=clean_literal.replace('@e','')
            tail= ' <T> '+clean_literal
        formatted_triples.append(head+relation+tail)
    return formatted_triples

def process_triples_from_ranking(present_, topk):
    df = pd.read_csv(present_, delimiter=', \'', quotechar='"', header=None, index_col=None, engine="python")
    #print(df.head())
    df = df.map(lambda x: x.replace("(", "").replace(")", "").replace("'", "") if x is not None else x)
    df['triples'] = df.apply(lambda x: '\t'.join(x.astype(str)), axis=1)
    triples=df['triples'].tolist()
    formatted_triples=format_triples(triples, topk)
    return formatted_triples

def save_triples(formatted_triples, fname):
    with open(fname, 'w') as output_file:
        for triple in formatted_triples:
            output_file.write(f"{triple}")
        output_file.close()

def is_non_zero_file(fpath):  
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def formatting_output(system_, dataset, base_model, dir_, topk):
    root_path_verbalization = f"../../data/{dataset}/predictions/{base_model}/{dir_}"
    system_dir = f"{root_path_verbalization}/{system_}"
    triples_formatted_dir=f'{system_dir}/triples-formatted/'
    if not os.path.exists(system_dir):
        os.makedirs(system_dir)
    if not os.path.exists(triples_formatted_dir):
        os.makedirs(triples_formatted_dir)
    triples_all_dir= f'../../data/{dataset}/predictions/{base_model}/{dir_}/{system_}/ranking'
    triples_all_list = glob.glob(triples_all_dir + "/*") 
    for ename in triples_all_list:
        #print(ename)
        triples=process_triples_from_ranking(ename, topk)
        formatted_triples= triples
        fname=ename.split('/')[-1]
        save_triples(formatted_triples, f"{triples_formatted_dir}/{fname}")

def main(args):
    if args.dataset=="ESSUM-DBpedia":
        dataset = "ESBM-DBpedia"
    elif args.dataset=="ESSUM-FACES":
        dataset = "FACES"
    else:
        raise NotImplementedError
    #print(dataset)
    # Set directories
    dir_ = "semantic-constraints" if args.semantic_constraints else "non-semantic-constraints"
    triples_ranking_dir = f"../../data/{dataset}/predictions/{args.base_model}/{dir_}/{args.combined_model}/ranking/"
    os.makedirs(triples_ranking_dir, exist_ok=True)

    # Fetch files
    kge_files = glob.glob(f"../../data/{dataset}/predictions/KGE/{dir_}/{args.kge_model}/ranking/*")
    gpt_files = glob.glob(f"../../data/{dataset}/predictions/LLM/{args.gpt_model}/triples-selected/*")

    for num, file in enumerate(kge_files):
        f = open(file, "r")
        filename = file.split("/")[-1]
        entity_name = filename.replace(".txt", "").strip()
        print(entity_name, f" generating triples-ranking {num+1}/{len(kge_files)} ")
        if os.path.isfile(f"{triples_ranking_dir}/{filename}"):
            continue
        kge_triples = f.readlines()
        f.close()
        triples = []
        for triple in kge_triples:
            triple = triple.replace("\n", "")
            triple = triple.split("\t")
            h1 = triple[0].strip()
            r1 = triple[1].strip()
            t1 = triple[2].strip()
            triples.append((h1, r1, t1))
            #print(h1,r1,t1)
        f = open(f"../../data/{dataset}/predictions/LLM/{args.gpt_model}/triples-selected/{filename}")
        gpt_triples = f.readlines()
        for triple in gpt_triples:
            triple = triple.replace("\n", "")
            triple = triple.split("\t")
            #print(triple)
            h1 = triple[0].strip()
            r1 = triple[1].strip().replace(" ", "")
            t1 = triple[2].strip()
            if t1 == "":
                continue
            triples.append((h1, r1, t1))
        f.close()
    
        triple_scoring = {}
        for triple in triples:
            h, r, t = triple
            if r == "http://dbpedia.org/ontology/abstract" :
                continue
            score = fetch_triple_score(r)
            triple_scoring[triple] = score
        # Sort and write top-k triples
        sorted_triples = sorted(triple_scoring.items(), key=lambda item: item[1], reverse=True)[:args.topk]
        with open(f"{triples_ranking_dir}/{filename}", "w") as f:
            for triple, score in sorted_triples:
                f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")
        formatting_output(args.combined_model, dataset, args.base_model, dir_, args.topk)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank triples based on relevance in DBpedia")
    parser.add_argument("--kge_model", type=str, default="conve_text", help="Knowledge Graph Embedding model name")
    parser.add_argument("--gpt_model", type=str, default="gpt-4", help="GPT model name")
    parser.add_argument("--dataset", type=str, default="ESBM-DBpedia", help="Dataset name")
    #parser.add_argument("--llm_dataset", type=str, default="ESBM-DBpedia", help="Dataset used for LLM")
    parser.add_argument("--combined_model", type=str, default="conve_text_gpt-4", help="Combination model name")
    parser.add_argument("--base_model", type=str, default="ANTS", help="Base model name")
    parser.add_argument("--topk", type=int, default=20, help="Number of top triples to select")
    parser.add_argument("--semantic_constraints", action='store_true', help="Enable semantic constraints")
    
    
    args = parser.parse_args()
    main(args)
