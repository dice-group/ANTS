import os
import time
import argparse
from openai import OpenAI
import json

client = OpenAI(
    # This is the default and can be omitted
    api_key="<your api key>",
)

def query_llm(prompt):
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
        model="gpt-4-0125-preview",
        #model="gpt-3.5-turbo",
    )
    return response.choices[0].message.content

def main(args):
    # Define the domain and scope
    if args.dataset=="ESSUM-DBpedia":
        dataset = "ESBM-DBpedia"
    else:
        dataset = "FACES"
    model = args.system
    domain = "DBpedia properties"
    scope = "Common DBpedia properties for an entity"
    #fr = open("data/ESBM/esbm-elist.txt", "r")
    fr = open(f"../../data/{dataset}/elist.txt", "r")
    content = fr.readlines()
    count = 0
    for num, item in enumerate(content):
        if num==0:
            continue
        print(f"Generating relations {num+1}/{len(content)}")
        item = item.replace("\n", "")
        item = item.split("\t")
        eid = item[0]
        entity = item[1]
        euri = item[2]
        entity_name = entity.replace("_", " ")
        
        if os.path.isfile(f"../../data/{dataset}/relevant-triples/LLM/{model}/relations/{entity}.txt"):
            count = 3
            continue
        if count == 3:
            count = 0
            time.sleep(60)
        # Query the LLM for entities in the domain
        entity_prompt = f"Generate the top 20 most relevant and diversified DBpedia attributes (relation names) for the entity {entity_name} , ignoring abstract, remark, description, thumbnail, and type. The outputs should only include a list of URI properties (e.g., http://dbpedia.org/ontology/[relation_name]) that provide an entity or class as objects. There is no requirement for an explanation or ordered numbers in the list; any opening and closing explanation"
        print(entity_prompt)
        
        triples = query_llm(entity_prompt)
        #print(triples)
        
        # Save to file 
        f = open(f"../../data/{dataset}/relevant-triples/LLM/{model}/relations/{entity}.txt", "w")
        f.write(triples)
        f.close()
        count +=1
    fr.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and format triples")
    parser.add_argument("--system", type=str, required=True, help="System name (e.g., gpt-3.5)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., FACES)")
    args = parser.parse_args()
    main(args)

