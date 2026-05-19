import json
import os
import tempfile
import gurobipy as gp
import time
import networkx as nx
import numpy as np
from typing import Optional

from evaluate._utils import (
    wltest_coloring_two,
    graph_generator,
    check_color_equivalence, derive_adjacency
)
from evaluate.python_executor import PythonExecutor

# import SD checking tools
from evaluate._utils import check_symmetric_decomposable, check_wl_determinable

# import lp generation tools
from evaluate._lp_utils import ensure_imports, process_code_for_lp

import evaluate.equivalence_check as equivcheck


def read_sample(problem_id):
    with open(f"data/sample_{problem_id}.json", "r") as f:
        data = json.load(f)
    return data


def convert_to_lp(
    model_name: str, data_dir: str, code: str, data_path: str, problem_id: Optional[int] = None, verbose: bool = False
):
    """
    Convert Gurobi Python code to LP file

    Args:
        code: Gurobi Python code
        data_path: Path to the sample data JSON file
        problem_id: Sample ID, used to generate file name
        verbose: Whether to print detailed information
    Returns:
        tuple: (Whether successful, LP file content or error message, LP file path)
    """
    # Create temporary directory to store LP files
    data_dir_name = data_dir.split("/")[-1]
    temp_dir = f'temp_lp/{model_name}_{data_dir_name}_lp/'#tempfile.mkdtemp() #'/home/ziweizhu/ziwei/or_modeling_tianding/temp_lp/'
    os.makedirs(temp_dir, exist_ok=True)
    if problem_id is not None:
        lp_file_name = f"{problem_id}_model_{data_path.split('/')[-1].split('.')[0]}.lp"
        lp_file_path = os.path.join(temp_dir, lp_file_name)
    else:
        lp_file_path = os.path.join(temp_dir, "model_test.lp")
    
    # If the LP file exists, delete it before writing new content
    if os.path.exists(lp_file_path):
        os.remove(lp_file_path)

    if verbose:
        print(f"Processing code for to write LP file at: {lp_file_path}")


    code_for_lp = process_code_for_lp(code, data_path, lp_file_path)
    if verbose:
        print(f"*" * 10, "Processed code", "*" * 10)
        #print(code_for_lp)
    # This is not necessary for RL training
    code_for_lp = ensure_imports(code_for_lp)

    os.makedirs(f"temp_code/{model_name}/", exist_ok=True)
    with open(f"temp_code/{model_name}/temp_exec_{problem_id}.py", "w", encoding="utf-8") as f:
        f.write(code_for_lp)

    import subprocess
    result = subprocess.run(["python", f"temp_code/{model_name}/temp_exec_{problem_id}.py"], capture_output=True, text=True)
    #print("Output:", result.stdout)
    #print("Error:", result.stderr)  

    if os.path.exists(lp_file_path):
        with open(lp_file_path, "r") as f:
            lp_content = f.read()
        return True, lp_content, lp_file_path
    else:
        return False, "LP file was not created due to error: " + result.stderr, None




def store_sample(
    code,
    problem_id,
    reference_lp_path,
    problem_class,
    domain,
    is_equivalent,
    equivalence_error,
    output_dir="/outputs/bench4opt",
):
    """
        Store the sample data in a JSON file for later evaluation and analysis

    Args:
        code: Gurobi Python code
        problem_id: Sample ID (e.g., LP_001, MILP_002)
        path: Sample file path
        problem_class: Problem class (e.g., Network Flow)
        domain: Application domain (e.g., Health Care, Transportation)
        output_dir: Output directory, default is 'data_output'
    """
    # Create sample data dictionary
    sample_data = {
        "sample_id": problem_id,
        "sample_code": code,
        "sample_path": f"data/sample_{problem_id}.json",
        "reference_lp_path": reference_lp_path,
        "problem_type": problem_id.split("_")[0],  # Extract problem type from ID (LP/MILP)
        "is_equivalent": is_equivalent,
        "equivalence_error": equivalence_error,
        "problem_class": problem_class,
        "domain": domain,
        # "code":,
        # "reward":,
        # "verification":
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate output file path, consistent with the path used in read_sample function
    output_file = os.path.join(output_dir, f"sample_{problem_id}.json")

    # Save as JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)

    print(f"Sample {problem_id} saved to {output_file}")




def evaluate_code_re(
    model_name: str, data_dir: str, code: str, data_path: str, reference_lp_path: str, verbose: bool = False, problem_id: Optional[int] = None
) -> dict:
    """
    Evaluate LLM-generated Gurobi Python code using check_lp_equivalence from equivalence_check.py

    Args:
        code: LLM-generated Gurobi Python code
        data_path: Sample data path
        reference_lp_path: Reference LP file path
        verbose: Whether to print detailed information
        problem_id: Optional problem ID for generating file names

    Returns:
        tuple: (code_eval_result, equivalence_eval_result)
            code_eval_result: Code execution result dictionary
            equivalence_eval_result: Equivalence check result dictionary
    """
    # Check if the reference LP file exists
    if not reference_lp_path or not os.path.exists(reference_lp_path):
        code_eval_result = {
            "success": False,
            "message": "no reference lp",
            "lp_file_path": "",
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # Convert code to LP file

    code_success, code_result, lp_file_path = convert_to_lp(
        model_name, data_dir, code, data_path, verbose=verbose, problem_id=problem_id
    )

    if not code_success:
        code_eval_result = {
            "success": False,
            "message": code_result,
            "lp_file_path": lp_file_path,
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # Use check_lp_equivalence from equivalence_check.py for equivalence check
    start_time = time.time()
    is_equivalent, error_msg,time_info = equivcheck.check_lp_equivalence(reference_lp_path, lp_file_path)
    end_time = time.time()
    total_equivalence_check_time = end_time - start_time
    equivalence_check_time = {
        "total_time": total_equivalence_check_time,
        "step_time": time_info
    }

    code_eval_result = {
        "success": True,
        "message": code_result,
        "lp_file_path": lp_file_path,
    }
    
    equivalence_eval_result = {
        "success": is_equivalent,
        "message": error_msg
    }

    return code_eval_result, equivalence_eval_result, equivalence_check_time



def evaluate_code_solver(
    model_name: str, data_dir: str, code: str, data_path: str, reference_lp_path: str, verbose: bool = False, problem_id: Optional[int] = None
) -> dict:
    """
    Evaluate LLM-generated Gurobi Python code using solver for equivalence check

    Args:
        code: LLM-generated Gurobi Python code
        data_path: Sample data path
        reference_lp_path: Reference LP file path
        verbose: Whether to print detailed information
        problem_id: Optional problem ID for generating file names

    Returns:
        tuple: (code_eval_result, equivalence_eval_result)
            code_eval_result: Code execution result dictionary
            equivalence_eval_result: Equivalence check result dictionary
    """
    # Check if the reference LP file exists
    if not reference_lp_path or not os.path.exists(reference_lp_path):
        code_eval_result = {
            "success": False,
            "message": "no reference lp",
            "lp_file_path": "",
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # Convert code to LP file

    code_success, code_result, lp_file_path = convert_to_lp(
        model_name, data_dir, code, data_path, verbose=verbose, problem_id=problem_id
    )

    if not code_success:
        code_eval_result = {
            "success": False,
            "message": code_result,
            "lp_file_path": lp_file_path,
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # Use solver_evaluation for equivalence check
    start_time = time.time()
    is_equivalent,msg,eval_time = solver_evaluation(reference_lp_path, lp_file_path)
    end_time = time.time()
    total_equivalence_check_time = end_time - start_time
    equivalence_check_time = {
        "total_time": total_equivalence_check_time
    }

    code_eval_result = {
        "success": True,
        "message": code_result,
        "lp_file_path": lp_file_path,
    }
    
    equivalence_eval_result = {
        "success": is_equivalent,
        "message": msg
    }

    return code_eval_result, equivalence_eval_result, equivalence_check_time


def solver_evaluation(lp_path1, lp_path2):
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    
    try:
        model1 = gp.read(lp_path1,env)
        model2 = gp.read(lp_path2,env)
    except Exception as e:
        return False, f"Error reading LP file: {e}", 0
        
    start_time = time.time()
    try:
        model1.optimize()
    except Exception as e:
        return False, f"Error optimizing model: {e}", 0

    try:
        model2.optimize()
    except Exception as e:
        return False, f"Error optimizing model: {e}", 0
    
    # Status mapping dictionary
    status_map = {
        gp.GRB.LOADED: "Model loaded but not solved",
        gp.GRB.OPTIMAL: "optimal",
        gp.GRB.INFEASIBLE: "infeasible",
        gp.GRB.INF_OR_UNBD: "infeasible or unbounded",
        gp.GRB.UNBOUNDED: "unbounded",
        gp.GRB.CUTOFF: "cutoff",
        gp.GRB.ITERATION_LIMIT: "iteration limit",
        gp.GRB.NODE_LIMIT: "node limit",
        gp.GRB.TIME_LIMIT: "time limit",
        gp.GRB.SOLUTION_LIMIT: "solution limit",
        gp.GRB.INTERRUPTED: "interrupted",
        gp.GRB.NUMERIC: "numeric",
        gp.GRB.SUBOPTIMAL: "suboptimal",
        gp.GRB.INPROGRESS: "in progress",
        gp.GRB.USER_OBJ_LIMIT: "user objective limit"
    }
    
    # Check reference model status
    if model1.status != gp.GRB.OPTIMAL:
        return False, f"reference model status: {status_map.get(model1.status, 'unknown status')}", 0
    
    try:
        obj1 = model1.getAttr("ObjVal")
    except Exception as e:
        return False, f"cannot get reference model objective value: {e}", 0
    
    # Check evaluated model status
    if model2.status == gp.GRB.OPTIMAL:
        try:
            obj2 = model2.getAttr("ObjVal")
            eval_result = abs(obj1 - obj2) < 1e-6
            msg = "both models are optimal"
        except Exception as e:
            return False, f"optimal but cannot get objective value: {e}", 0
    else:
        eval_result = False
        msg = f"evaluated model status: {status_map.get(model2.status, 'unknown status')}"
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    env.dispose()
    return eval_result, msg, eval_time



def evaluate_code_ged(
    model_name: str, data_dir: str, code: str, data_path: str, reference_lp_path: str, verbose: bool = False, problem_id: Optional[int] = None
) -> dict:
    """
    Evaluate LLM-generated Gurobi Python code using solver for equivalence check

    Args:
        code: LLM-generated Gurobi Python code
        data_path: Sample data path
        reference_lp_path: Reference LP file path
        verbose: Whether to print detailed information
        problem_id: Optional problem ID for generating file names

    Returns:
        tuple: (code_eval_result, equivalence_eval_result)
            code_eval_result: Code execution result dictionary
            equivalence_eval_result: Equivalence check result dictionary
    """
    # Check if the reference LP file exists
    if not reference_lp_path or not os.path.exists(reference_lp_path):
        code_eval_result = {
            "success": False,
            "message": "no reference lp",
            "lp_file_path": "",
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # Convert code to LP file

    code_success, code_result, lp_file_path = convert_to_lp(
        model_name, data_dir, code, data_path, verbose=verbose, problem_id=problem_id
    )

    if not code_success:
        code_eval_result = {
            "success": False,
            "message": code_result,
            "lp_file_path": lp_file_path,
        }
        equivalence_eval_result = {"success": False, "message": ""}
        return code_eval_result, equivalence_eval_result, ""

    # Use solver_evaluation for equivalence check
    start_time = time.time()
    ged = calculate_code_ged(reference_lp_path, lp_file_path)
    end_time = time.time()
    total_equivalence_check_time = end_time - start_time
    equivalence_check_time = {
        "total_time": total_equivalence_check_time
    }

    code_eval_result = {
        "success": True,
        "message": code_result,
        "lp_file_path": lp_file_path,
    }
    


    return code_eval_result, ged, equivalence_check_time


def calculate_ged(G1, G2):
    """
    Calculate the normalized Graph Edit Distance (GED) between two graphs.
    
    Args:
        G1, G2: networkx graph objects with node and edge features
        
    Returns:
        float: Normalized Graph Edit Distance
    """
    def node_subst_cost(node1_data, node2_data):
        """Node substitution cost: L1 distance between feature vectors"""
        return np.sum(np.abs(node1_data['feature'] - node2_data['feature']))
    
    def edge_subst_cost(edge1_data, edge2_data):
        """Edge substitution cost: L1 distance between feature vectors"""
        return np.sum(np.abs(edge1_data['feature'] - edge2_data['feature']))
    
    def node_del_cost(node_data):
        """Node deletion cost: fixed at 1"""
        return 1.0
    
    def edge_del_cost(edge_data):
        """Edge deletion cost: fixed at 1"""
        return 1.0
    
    def node_ins_cost(node_data):
        """Node insertion cost: fixed at 1"""
        return 1.0
    
    def edge_ins_cost(edge_data):
        """Edge insertion cost: fixed at 1"""
        return 1.0
    
    # Calculate original Graph Edit Distance
    ged = nx.graph_edit_distance(G1, G2,
                               node_subst_cost=node_subst_cost,
                               edge_subst_cost=edge_subst_cost,
                               node_del_cost=node_del_cost,
                               edge_del_cost=edge_del_cost,
                               node_ins_cost=node_ins_cost,
                               edge_ins_cost=edge_ins_cost)
    
    # Normalize the Graph Edit Distance using the maximum size of the two graphs
    max_size = max(len(G1.nodes) + len(G1.edges), len(G2.nodes) + len(G2.edges))
    normalized_ged = ged / max_size if max_size > 0 else 0
    
    return 1-normalized_ged

def calculate_code_ged(lp_path1,lp_path2):
    with gp.Env(empty=True) as env:
        env = gp.Env(empty=True)
        env.setParam("LogToConsole", 0)
        env.start()
        model1 = gp.read(lp_path1,env)
        model2 = gp.read(lp_path2,env)
        
        # Standardize boundary constraints
        model1 = equivcheck.convert_boundary(model1)
        model1.write("model1_converted.lp")
        model2 = equivcheck.convert_boundary(model2)
        model2.write("model2_converted.lp")


        # Generate bipartite graphs
        G1 = equivcheck.generate_bipartite(model1)
        G2 = equivcheck.generate_bipartite(model2)
        # Calculate Graph Edit Distance
        dist = calculate_ged(G1, G2)
        #print(dist)
    env.close()
    return dist