import os
import glob
import argparse

def create_directories(evaluation_dir, system_dir):
    """
    Creates necessary directories if they don't exist.
    """
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    if not os.path.exists(system_dir):
        os.makedirs(system_dir)

def get_directories(system_, dataset, base_model, semantic_constraints):
    """
    Determine the correct directories based on input parameters.
    """
    dir_ = "/semantic-constraints" if semantic_constraints else "/non-semantic-constraints"
    if base_model in ["baselines", "LLM"]:
        dir_ = ""
    evaluation_dir = f"../data/{dataset}/predictions/{base_model}{dir_}/{system_}"
    system_dir = f"{evaluation_dir}/evaluation"
    
    return evaluation_dir, system_dir

def process_hypotheses(hyp_dir, system_dir):
    """
    Process hypothesis files by cleaning and aggregating them into a single file.
    """
    hyp_entities = sorted(glob.glob(f'{hyp_dir}/*'))
    print(f"Number of hypothesis entities: {len(hyp_entities)}")

    with open(f"{system_dir}/hyp.txt", "w") as hyp_file:
        for hyp_entity in hyp_entities:
            print(f"Processing: {hyp_entity}")
            with open(hyp_entity, "r") as file:
                content = file.read().replace("\n", "").replace("< pad >", '').replace("< / s >", '').strip()
                hyp_file.write(f"{content}\n")

def process_references(hyp_entities, system_dir, dataset):
    """
    Process reference files and aggregate them into a single file.
    """
    with open(f"{system_dir}/refs.txt", "w") as ref_file:
        for num, hyp_entity in enumerate(hyp_entities):
            entity_filename = hyp_entity.split("/")[-1].replace(".txt-verbalized.txt", "")
            print(f"Processing reference: {num + 1}, {entity_filename}")

            ref_file_path = f"../data/{dataset}/ESSUM/{entity_filename}"
            with open(ref_file_path, "r") as file:
                content = file.readlines()
                sentences = " ".join([line.replace("\n", "").replace("< pad >", "").replace("< / s >", "").replace("<s>", "").strip() for line in content]).strip()
                ref_file.write(f"{sentences}\n")

def main(system_, dataset, base_model, semantic_constraints):
    evaluation_dir, system_dir = get_directories(system_, dataset, base_model, semantic_constraints)
    create_directories(evaluation_dir, system_dir)
    
    hyp_dir = f"{evaluation_dir}/outputs/"
    process_hypotheses(hyp_dir, system_dir)
    
    hyp_entities = sorted(glob.glob(f'{hyp_dir}/*'))
    process_references(hyp_entities, system_dir, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument('--system', type=str, default="gpt-3.5", help="System name (e.g., gpt-3.5)")
    parser.add_argument('--dataset', type=str, default="FACES", help="Dataset name (e.g., FACES)")
    parser.add_argument('--base_model', type=str, default="LLM", help="Base model name (e.g., LLM or baselines)")
    parser.add_argument('--semantic_constraints', action='store_true', help="Use this flag to apply semantic constraints")

    args = parser.parse_args()

    main(args.system, args.dataset, args.base_model, args.semantic_constraints)
