import os
import glob
import torch
import tqdm
import argparse

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from sacremoses import MosesTokenizer, MosesDetokenizer

# Initialize the tokenizer and model
def initialize_model(model_path, device):
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    config = T5Config.from_pretrained("t5-large")
    print(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)
    model.config.update(model.config.task_specific_params['translation_en_to_de'])
    model.to(device)
    return tokenizer, model

# Load triples from a file
def jload(fname):
    with open(fname, 'r') as input_file:
        triples = input_file.read().splitlines()
    return triples

# Prepare data for processing
def prep_data(fname, model_dir):
    test_raw = jload(fname)
    ename = fname.split("/")[-1]
    test = [('Graph to Text: ' + x.replace('_', ' '), None) for x in test_raw]
    torch.save(test, f'{model_dir}/{ename}.pt')

# Batch data for processing
def batch_it(datas, batch_size):
    return [datas[i:i + batch_size] for i in range(0, len(datas), batch_size)]

# Generate predictions for a batch
def pred_one(batch, model, tokenizer):
    model.eval()
    inp = tokenizer([x[0] for x in batch], return_tensors='pt', padding=True)['input_ids'].to(model.device)
    
    # Estimate the max_length in tokens
    avg_tokens_per_word = 1.5
    max_words = 379
    estimated_max_length = int(max_words * avg_tokens_per_word)
    
    pred = model.generate(
        input_ids=inp, 
        max_length=estimated_max_length,
        num_beams=4,
        repetition_penalty=2.0,
        do_sample=False,
        early_stopping=True
    )
    return tokenizer.batch_decode(pred)

# Evaluate the model and write outputs to a file
def eval_g2t(datas, model, tokenizer, demo_name):
    model.eval()
    hyp = []
    mt = MosesTokenizer(lang='en')
    md = MosesDetokenizer(lang='en')

    with tqdm.tqdm(batch_it(datas, 2)) as tqb:
        for batch in tqb:
            with torch.no_grad():
                pred = pred_one(batch, model, tokenizer)
            hyp.extend(pred)
    
    with open(demo_name, 'w') as wf_h:
        for h in hyp:
            wf_h.write(md.detokenize(mt.tokenize(str(h))) + '\n')

# Set up directories
def setup_directories(dataset, base_model, semantic_constraints, system_):
    dir_ = "/semantic-constraints" if semantic_constraints else "/non-semantic-constraints"
    if base_model in ["baselines", "LLM"]:
        dir_ = ""
    
    root_path = f"../../data/{dataset}/predictions/{base_model}{dir_}"
    system_dir = f"{root_path}/{system_}"
    triples_formatted_dir = f'{system_dir}/triples-formatted/'
    output_dir = f'{system_dir}/outputs'
    model_dir = f'{system_dir}/models'
    
    for directory in [system_dir, triples_formatted_dir, output_dir, model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    return triples_formatted_dir, output_dir, model_dir

# Process entities
def process_entities(triples_formatted_dir, output_dir, model_dir, model, tokenizer):
    entities_list = glob.glob(triples_formatted_dir + "*")
    
    for num, entity_file in enumerate(entities_list):
        print(entity_file, f"Processing entity {num + 1}/{len(entities_list)}")
        fname = entity_file.split('/')[-1]
        
        prep_data(entity_file, model_dir)
        load_processed_data = torch.load(f'{model_dir}/{fname}.pt')
        eval_g2t(load_processed_data, model, tokenizer, f'{output_dir}/{fname}-verbalized.txt')

# Main function to execute the script
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process and evaluate a model's predictions.")
    
    parser.add_argument('--system', type=str, default="gpt-3.5", help="System name")
    parser.add_argument('--dataset', type=str, default="ESSUM-FACES", help="Dataset name")
    parser.add_argument('--base_model', type=str, default="LLM", help="Base model name (e.g., baselines or LLM)")
    parser.add_argument('--semantic_constraints', type=bool, default=True, help="Use semantic constraints or not")
    parser.add_argument('--model_path', type=str, default="p2-verbalization-model.pt", help="Path to the model")
    
    args = parser.parse_args()
    if args.dataset=="ESSUM-DBpedia":
        dataset = "ESBM-DBpedia"
    elif args.dataset=="ESSUM-FACES":
        dataset = "FACES"
    else:
        raise NotImplementedError
    # Setup directories
    triples_formatted_dir, output_dir, model_dir = setup_directories(
        dataset, args.base_model, args.semantic_constraints, args.system
    )
    
    # Initialize model and tokenizer
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer, model = initialize_model(args.model_path, device)
    
    # Process entities
    print(triples_formatted_dir)
    process_entities(triples_formatted_dir, output_dir, model_dir, model, tokenizer)

if __name__ == "__main__":
    main()
