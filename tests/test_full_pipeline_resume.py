import os
import json
from pprint import pprint
from typing import Tuple, Dict, List, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from evaluate._lp_utils import extract_python_code
import evaluate.evaluate_code as evaluate_code
import time
import threading
import gurobipy

def get_sample(data_dir: str, sample_id: int) -> dict:
    """
    Load a sample from the data path by its ID.
    
    Args:
        data_dir: Path to the data directory
        sample_id: ID of the sample to retrieve
        
    Returns:
        dict: Sample data dictionary
    """
    data = [json.loads(line) for line in open(os.path.join(data_dir, 'test.jsonl'), 'r')]
    sample_data = data[sample_id]

    # sample_data is a dictionary with the following key-value pairs:
    sample_data = {
        "id": sample_data["id"],
        "data_path": sample_data["data_path"],     # LP问题的数据路径
        "problem": sample_data["problem"],         # LP问题的描述
        "reference_lp_path": sample_data["reference_lp_path"], # LP问题的参考解
        "problem_type": sample_data["problem_type"],      # LP/MILP
        "problem_class": sample_data["problem_class"],   # 问题类型
    }

    return sample_data


def generate_prompt(sample_data: dict) -> str:
    """
    Generate a prompt based on the sample data.
    
    Args:
        sample_data: Dictionary containing sample information
        
    Returns:
        dict: Updated sample data with prompt added
    """

    grb_version = str(gurobipy.gurobi.version())
    prompt = f"""You are an expert in optimization modeling and gurobipy. Build a model for the following optimization problem and implement it using Gurobi Optimizer with the `gurobipy` package version {grb_version}. The requirements are as follows:

1. Do not optimize or solve the model. Instead, save the model as an `.lp` file.
2. Specify the upper and lower bounds for all variables.
3. Be careful with the data types and dimensions of all variables and parameters.
4. Do not use any extra parameters that are not provided in the data file. If additional parameters are necessary, define them clearly.
5. Output the Python code only. Do not output anything else.
6. Make sure your output code can be run directly in Gurobi Optimizer and saves the desired `.lp` file."""
    problem = sample_data['problem']
    prompt = prompt + f"\n\n{problem}"

    return prompt

def generate_completion(sample_data: dict, model_name: str, **generation_kwargs) -> str:
    """
    Generate a completion using the specified model.
    
    Args:
        sample_data: Dictionary containing sample information and prompt
        model_name: Name of the model to use
        **generation_kwargs: Additional keyword arguments for generation
        
    Returns:
        dict: Updated sample data with completion added
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )

    kwargs = {
        "model": model_name,
        #"temperature": 0.0,
        #"top_p": 1.0,
        #"max_tokens": 16384,
        "seed": 42
    }
    for key, value in generation_kwargs.items():
        kwargs[key] = value
    
    request_output = client.chat.completions.create(
        messages=[{"role": "user", "content": sample_data['prompt']}],
        **kwargs
    )

    completion = request_output.choices[0].message.content
    return completion


def evaluate_completion(data_dir: str, sample_data: dict, model_name: str) -> Tuple[str, int, str]:
    """
    Evaluate the quality of the completion.
    
    Args:
        data_dir: Path to the data directory
        sample_data: Dictionary containing sample information and completion
        
    Returns:
        dict: Updated sample data with evaluation metrics
    """
    from evaluate._lp_utils import extract_python_code


    code = extract_python_code(sample_data["completion"])
    
    print("*"*10, "extracted code", "*"*10)
    #print(code)

    data_path = os.path.join(data_dir, sample_data["data_path"])
    reference_lp_path = os.path.join(data_dir, sample_data["reference_lp_path"])
    code_eval_result, wl_eval_result, equivalence_check_time = evaluate_code.evaluate_code_re(
                    model_name=model_name,
                    data_dir=data_dir,
                    code=code,
                    data_path=data_path,
                    reference_lp_path=reference_lp_path,
                    verbose=True,  # print detailed evaluation results
                    problem_id = sample_data['id'].split("_")[1]
                )

    if code_eval_result["success"]:
        code_success = True
        code_error_msg = ""
        code = code_eval_result["lp_file_path"]
        if wl_eval_result["success"]:
            wl_success = True
            wl_error_msg = wl_eval_result['message']
        else: 
            wl_success = False
            wl_error_msg = wl_eval_result['message']
    else:
        code_success = False
        code_error_msg = code_eval_result['message']
        code = code_eval_result["lp_file_path"]
        wl_success = False
        wl_error_msg = ""

    # Set reward for code generation
    code_reward = 1.0 if code_success else 0.0
    # Set reward for or modeling 
    wl_reward = 1.0 if wl_success else 0.0

    reward = {'code_reward':code_reward,"wl_reward":wl_reward}

    verification = {"code_verification":code_error_msg,
                    "wl_verification":wl_error_msg}
    # Update sample data with results
    return (
        code,
        reward,
        verification,
        equivalence_check_time
    )

def test_full_pipeline(data_dir, sample_data, model_name, save_path):
    """Test the full evaluation pipeline"""
    # Run the pipeline
    # generate prompt
    prompt = generate_prompt(sample_data)
    print("*"*10,"Test generate_prompt success","*"*10)
    sample_data['prompt'] = prompt
    # generate completion
    sample_data['completion'] = generate_completion(sample_data, model_name)
    # evaluate completion
    code, reward, verification, equivalence_check_time = evaluate_completion(data_dir, sample_data, model_name)
    print("*"*10,"Test evaluate_completion success","*"*10)
    sample_data['code'] = code
    sample_data['reward'] = reward
    sample_data['verification'] = verification
    sample_data['equivalence_check_time'] = equivalence_check_time
    return sample_data


def load_existing_results(save_path: str) -> Dict[str, dict]:
    """
    Load existing results file and return a dictionary of completed samples
    
    Args:
        save_path: Path to the results file
        
    Returns:
        Dict[str, dict]: Dictionary of completed samples, with sample_id as the key
    """
    if not os.path.exists(save_path):
        print(f"结果文件 {save_path} 不存在，将从头开始")
        return {}
    
    try:
        with open(save_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Convert results to a dictionary with sample_id as the key
        completed_samples = {}
        for result in results:
            if 'id' in result:
                completed_samples[result['id']] = result
        
        print(f"Successfully loaded {len(completed_samples)} completed results")
        return completed_samples
    
    except Exception as e:
        print(f"Error loading results file: {e}")
        return {}


def save_results_incremental(all_results: List[dict], save_path: str, lock: threading.Lock):
    """
    Incrementally save results to a file
    
    Args:
        all_results: List of all results
        save_path: Path to save the results
        lock: Thread lock
    """
    with lock:
        try:
            # Create a backup
            if os.path.exists(save_path):
                backup_path = save_path + ".backup"
                os.rename(save_path, backup_path)
            
            # Save results to file
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            
            # delete backup after successful save
            if os.path.exists(save_path + ".backup"):
                os.remove(save_path + ".backup")
                
            print(f"Results have been saved to {save_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
            # Restore backup
            if os.path.exists(save_path + ".backup"):
                os.rename(save_path + ".backup", save_path)


def get_completed_sample_ids(completed_samples: Dict[str, dict]) -> Set[str]:
    """
    Get the set of completed sample IDs
    
    Args:
        completed_samples: Dictionary of completed samples
        
    Returns:
        Set[str]: Set of completed sample IDs
    """
    return set(completed_samples.keys())


def filter_pending_samples(full_data: List[dict], completed_sample_ids: Set[str]) -> List[dict]:
    """
    Filter out pending samples
    
    Args:
        full_data: List of all sample data
        completed_sample_ids: Set of completed sample IDs
        
    Returns:
        List[dict]: List of pending samples
    """
    pending_samples = []
    for sample in full_data:
        if sample['id'] not in completed_sample_ids:
            pending_samples.append(sample)
    
    print(f"Total samples: {len(full_data)}, Completed: {len(completed_sample_ids)}, Pending: {len(pending_samples)}")
    return pending_samples


def merge_results(existing_results: Dict[str, dict], new_results: List[dict]) -> List[dict]:
    """
    Merge existing results with new results
    
    Args:
        existing_results: Dictionary of existing results
        new_results: List of new results
        
    Returns:
        List[dict]: Merged list of results
    """
    # Add new results to existing results
    for result in new_results:
        existing_results[result['id']] = result
    
    # Sort by original order
    all_results = []
    for sample_id in sorted(existing_results.keys()):
        all_results.append(existing_results[sample_id])
    
    return all_results


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test with a specific model (with resume support)")
    parser.add_argument("--model_name", type=str, default='gpt-4o-mini', help="Name of the tested model")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--save_path", type=str, help="path to save log files")
    parser.add_argument("--save_interval", type=int, default=10, help="save results every N batches")
    parser.add_argument("--data_dir", type=str, default="/Users/zhuziwei/Downloads/ICLR_exp_副本/data/bench4opt_mix", help="path to data directory")
    
    args = parser.parse_args()
    batch_size = args.batch_size
    save_interval = args.save_interval
    
    # check save_path
    if not args.save_path:
        print("Error: --save_path argument is required")
        exit(1)
    
    # create save directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # load existing results and get completed sample IDs
    existing_results = load_existing_results(args.save_path)
    completed_sample_ids = get_completed_sample_ids(existing_results)
    
    # Prepare all sample data
    full_data = []
    data_dir = args.data_dir
    
    for i in range(394):
        full_data.append(get_sample(data_dir, i))
    
    # Filter out pending samples
    pending_samples = filter_pending_samples(full_data, completed_sample_ids)
    
    if not pending_samples:
        print("All samples have been evaluated!")
        exit(0)
    
    # Initialize result lists
    all_results = list(existing_results.values())
    new_results = []
    
    # Create a thread lock for safe saving
    save_lock = threading.Lock()
    
    # Process pending samples
    total_batches = (len(pending_samples) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(pending_samples), batch_size), desc="Processing batches"):
        batch_data = pending_samples[batch_idx:batch_idx + batch_size]
        
        try:
            with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
                # Process current batch
                results = list(
                    executor.map(
                        lambda p: test_full_pipeline(data_dir, p, args.model_name, args.save_path),
                        batch_data
                    )
                )
                new_results.extend(results)
                all_results.extend(results)
                
                print(f"Batch {batch_idx//batch_size + 1}/{total_batches} completed, processed {len(results)} samples")
                
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Even if an error occurs, save the results that have been processed so far
            pass
        
        # Periodically save results
        if (batch_idx // batch_size + 1) % save_interval == 0:
            save_results_incremental(all_results, args.save_path, save_lock)
            print(f"Intermediate results saved, {len(all_results)} samples completed so far")
    
    # Save final results after processing all batches
    save_results_incremental(all_results, args.save_path, save_lock)
    print(f"Evaluation completed! A total of {len(all_results)} samples were processed, and the results have been saved to {args.save_path}")
