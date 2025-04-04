import os
import time
import argparse
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="<your api key>",
)

def query_llm(prompt, model):
    """
    Send a prompt to the LLM and return the response.
    """
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    return response.choices[0].message.content

def main(model, system, dataset):
    # Define the domain and scope
    domain = "DBpedia Triples"
    scope = "Generate triples based on subject and relations"
    if dataset=="ESSUM-DBpedia":
        dataset = "ESBM-DBpedia"
    else:
        dataset = "FACES"
    model=system
    fr = open(f"../data/{dataset}/elist.txt", "r")
    content = fr.readlines()
    count = 0
    for num, item in enumerate(content):
        if num==0:
            continue
        print(f"Generating missing literals {num}/{len(content)-1}")
        item = item.replace("\n", "")
        item = item.split("\t")
        eid = item[0]
        entity = item[1]
        euri = item[2]
        entity_name = entity.replace("_", " ")
        
        if os.path.isfile(f"../../data/{dataset}/relevant-triples/LLM/{model}/triples/{entity}.txt"):
            count = 3
            continue
        if count == 3:
            count = 0
            time.sleep(60)
            
        f = open(f"../../data/{dataset}/relevant-triples/LLM/{model}/relations/{entity}.txt", "r")
        relations = f.readlines()
        f.close()
        missing_triples = []
        for relation in relations:
            relation = relation.replace("\n", "")
            relation_name = relation.split("/")[-1]
            # Query the LLM for entities in the domain
            entity_prompt = f"""
            Complete the RDF triples for the given DBpedia entity. Use reliable external data sources for information retrieval to ensure accuracy. 
            The output should consist of fully complete triples without any additional explanation.

            Input Entity Name: "{entity_name}"
            RDF Triple Format: ($entity_name; $relation_name; ?)
            Relation: "{relation_name}"
            
            """
            triples = query_llm(entity_prompt, model)
            
            missing_triples.append(triples)

        # Save to file 
        if len(missing_triples)>0:
            f = open(f"../../data/{dataset}/predictions/LLM/{model}/triples/{entity}.txt", "w")
            for triple in missing_triples:
                f.write(f"{triple}\n")
            f.close()
        count +=1
    fr.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RDF triples using an LLM.")
    parser.add_argument("--model", type=str, required=True, help="Specify the LLM model to use (e.g., 'gpt-4-0125-preview').")
    parser.add_argument("--system", type=str, required=True, help="System name (e.g., gpt-3.5)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., ESSUM-DBpedia)")
    
    args = parser.parse_args()
    main(args.model, args.system, args.dataset)
