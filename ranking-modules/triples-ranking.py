import argparse
import os
import glob
import time
from http.client import RemoteDisconnected
from SPARQLWrapper import SPARQLWrapper, JSON

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

def main(args):
    # Set directories
    dir_ = "semantic-constraints" if args.semantic_constraints else "non-semantic-constraints"
    triples_ranking_dir = f"../data/{args.dataset}/predictions/{args.base_model}/{dir_}/{args.combine_model}/ranking/"
    os.makedirs(triples_ranking_dir, exist_ok=True)

    # Fetch files
    kge_files = glob.glob(f"../data/{args.dataset}/predictions/KGE/{dir_}/{args.kge_model}/ranking/*")
    gpt_files = glob.glob(f"../data/{args.llm_dataset}/predictions/LLM/{args.gpt_model}/triples-selected/*")

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
        f = open(f"../data/{args.llm_dataset}/predictions/LLM/{args.gpt_model}/triples-selected/{filename}")
        gpt_triples = f.readlines()
        for triple in gpt_triples:
            triple = triple.replace("\n", "")
            triple = triple.split("\t")
            print(triple)
            h1 = triple[0].strip()
            r1 = triple[1].strip().replace(" ", "")
            t1 = triple[2].strip()
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
        sorted_triples = sorted(triple_scoring.items(), key=lambda item: item[1], reverse=True)[:20]
        with open(f"{triples_ranking_dir}/{filename}", "w") as f:
            for triple, score in sorted_triples:
                f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank triples based on relevance in DBpedia")
    parser.add_argument("--kge_model", type=str, default="conve_text", help="Knowledge Graph Embedding model name")
    parser.add_argument("--gpt_model", type=str, default="gpt-4", help="GPT model name")
    parser.add_argument("--dataset", type=str, default="ESBM-DBpedia", help="Dataset name")
    parser.add_argument("--llm_dataset", type=str, default="ESBM-DBpedia", help="Dataset used for LLM")
    parser.add_argument("--combine_model", type=str, default="conve_text_gpt-4", help="Combination model name")
    parser.add_argument("--base_model", type=str, default="ANTS", help="Base model name")
    parser.add_argument("--semantic_constraints", action='store_true', help="Enable semantic constraints")
    
    args = parser.parse_args()
    main(args)
