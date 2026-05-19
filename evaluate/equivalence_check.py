import json
import os
import gurobipy as gp
import numpy as np
from gurobipy import GRB
import networkx as nx
import hashlib
from itertools import chain
import time


def float_array_to_str(arr, precision=6):
    """
    将浮点数组转为字符串, 保留一定小数位, 例如 [0.123456, 0.98765] -> "0.1235,0.9877"
    """
    return ",".join([f"{x:.{precision}f}" for x in arr.flatten()])

def hash_multiset(color, color_pairs):
    """
    对多重集合 {(x1,y1), (x2,y2), ...} 做哈希。
    - 顺序无关(排序)
    - 重复计数敏感(相同元素出现多次就会重复出现)
    """
    # 每个 (color, edge_attr) -> "color|edge_str"
    pair_strs = []
    for (c, e) in color_pairs:
        pair_strs.append(f"{c}|{e}")
    # sorting then combine into a single string
    pair_strs.sort()
    combined = "|".join(pair_strs)
    combined = f"{color}|{combined}"
    return hashlib.md5(combined.encode()).hexdigest()


def wl_test_nx(G):
    """
    conduct WL test on a networkx graph G, where node features are in 'feature' attribute and edge features are in 'feature' attribute of edges. The function returns the final node color partition and the number of iterations until convergence.
    
    Args:
        G: networkx graph, node features are numpy arrays of length 2, edge features are numpy arrays of length 1
        
    Returns:
        colors: dictionary of final node colors
        n_iter: number of iterations until convergence
    """

    max_iter = len(G.nodes)
    
    # Initialize node colors by converting node features to strings
    for node, feat in nx.get_node_attributes(G, 'feature').items():
        G.nodes[node]['color'] = float_array_to_str(feat)
    
    # Initialize edge colors by converting edge features to strings
    for u, v, data in G.edges(data=True):
        G[u][v]['color'] = float_array_to_str(data['feature'])
    
    # Record the color distribution after each iteration
    prev_partition = {}
    n_iter = 0
    
    # Iterate until convergence or reaching the maximum number of iterations
    for i in range(max_iter):
        new_partition = {}
        # For each node
        for node in G.nodes:
            # Get the color and edge feature pairs of neighbors
            neighbor_color_pairs = []
            for neighbor in G.neighbors(node):
                # Get neighbor color and edge color
                neighbor_color = G.nodes[neighbor]['color']
                edge_color = G[node][neighbor]['color']
                # Combine neighbor color and edge color
                neighbor_color_pairs.append((neighbor_color, edge_color))
            
            # Compute new color using hash_multiset
            node_color = G.nodes[node]['color']
            hashed_color = hash_multiset(node_color,neighbor_color_pairs)
            if hashed_color not in new_partition.keys():
                new_partition[hashed_color] = [node]
            else:
                new_partition[hashed_color].append(node)
        
        # Update colors in the graph
        for hashed_color, nodes in new_partition.items():
            for node in nodes:
                # Update node color attribute
                G.nodes[node]['color'] = hashed_color
        
        # Check for convergence (no increase in the number of color classes)
        if len(new_partition) == len(prev_partition):
            # If the same, return directly
            return new_partition, n_iter
        else:
            # If different, update prev_partition
            prev_partition = new_partition.copy()
            n_iter += 1
        
    return new_partition, n_iter

def convert_cons2boundary(model):
    nvars = model.NumVars
    vars = model.getVars()
    existing_constrs = model.getConstrs()
    cons_to_remove = []
    for n in range(nvars):
        var = vars[n]
        # Extract constraints that only include var
        existing_constrs_only_var = []
        for constr in existing_constrs:
            other_vars = [v for v in vars if v is not var and model.getCoeff(constr, v) != 0]
            if not other_vars and model.getCoeff(constr, var) != 0:
                existing_constrs_only_var.append(constr)
        #print(f'existing_constrs_only_var: {existing_constrs_only_var}')
        var_lb =[]
        var_ub=[]
        for constr in existing_constrs_only_var:
            if constr.Sense == GRB.GREATER_EQUAL:
                var_lb.append(constr.RHS/model.getCoeff(constr, var))
                cons_to_remove.append(constr)
            elif constr.Sense == GRB.LESS_EQUAL:
                var_ub.append(constr.RHS/model.getCoeff(constr, var))
                cons_to_remove.append(constr)
            elif constr.Sense == GRB.EQUAL:
                var_lb.append(constr.RHS/model.getCoeff(constr, var))
                var_ub.append(constr.RHS/model.getCoeff(constr, var))
                cons_to_remove.append(constr)
            else:
                raise NotImplementedError('Only implemented >=, <=, and =')
        #print(f'var_lb: {var_lb}, var_ub: {var_ub}')
        # If var_lb and var_ub exist, set var's LB and UB to the min and max of var_lb and var_ub
        if len(var_ub) > 0:
            var.setAttr(GRB.Attr.UB, min(var_ub))
        else:
            var.ub = var.ub
        if len(var_lb) > 0:
            var.setAttr(GRB.Attr.LB, max(var_lb))
        else:
            var.lb = var.lb
        
    for constr in cons_to_remove:
        model.remove(constr)
    model.update()

    return model

def convert_boundary(model):
    nvars = model.NumVars
    vars = model.getVars()
    # Extract variable names from vars
    var_names = [var.VarName for var in vars]
    
    for n in range(nvars):
        var = vars[n]
        obj_coeff = var.Obj
        lb = var.LB
        ub = var.UB
        
        existing_constrs = model.getConstrs()
        has_lb_constr = any(
            abs(model.getCoeff(constr, var) - 1.0) < 1e-6 and 
            abs(constr.RHS - lb) < 1e-6 and 
            constr.Sense == ">" 
            for constr in existing_constrs
        )
        has_ub_constr = any(
            abs(model.getCoeff(constr, var) - 1.0) < 1e-6 and 
            abs(constr.RHS - ub) < 1e-6 and 
            constr.Sense == "<" 
            for constr in existing_constrs
        )
        
        # Only add new constraints if they do not already exist
        if not has_lb_constr:
            model.addConstr(var >= lb)
        if ub != float('inf'):
            model.addConstr(var <= ub)
            
        model.update()

    # Remove the default bounds of Gurobi variables
    for var in vars:
        var.setAttr(GRB.Attr.LB, -GRB.INFINITY)
        var.setAttr(GRB.Attr.UB, GRB.INFINITY)

    model.update()
    return model

def generate_bipartite(model):
    # Create graph
    G = nx.Graph()

    # Create variable nodes and extract variable node features
    nvars = model.NumVars
    vars = model.getVars()
    var_names = [var.VarName for var in vars]
    for n, var in enumerate(vars):
        var = vars[n]
        obj_coeff = var.Obj
        if var.VType == GRB.BINARY:
            bin = 1.0
        elif var.VType==GRB.INTEGER:
            bin = 2.0
        else:
            bin = 0.0
        # Add node
        feature = [float(obj_coeff), bin, var.LB, var.UB]
        G.add_node(n, feature = np.array(feature, dtype=np.float32))
    
    #   Create constraint nodes and extract constraint node features, and add edges with edge features
    ncons = model.NumConstrs
    cons = model.getConstrs()

    for m, con in enumerate(cons):
        sense = con.Sense
        rhs = con.RHS
        
        if sense == GRB.EQUAL:
            cons_sense = 0.0
        elif sense == GRB.LESS_EQUAL:
            cons_sense = -1.0
        elif sense == GRB.GREATER_EQUAL:
            cons_sense = 1.0
        else:
            raise NotImplementedError('Only implemented feature >=,<=, and, =')
        
        feature = [float(rhs), cons_sense]
        G.add_node(nvars + m, feature = np.array(feature, dtype=np.float32))
        
        # Get all variables in the constraint
        row = model.getRow(con)
        
        # Iterate over variables in the constraint
        for i in range(row.size()):
            # Get the index of the variable in the vars list
            var_idx = var_names.index(row.getVar(i).varName)
            # Get the coefficient
            coeff = row.getCoeff(i)
            # Add edge, edge feature is the coefficient
            G.add_edge(var_idx, nvars + m, feature=np.array([coeff], dtype=np.float32))

    return G


def check_wl_determinable(same_color_partition):
    """
    Check if the WL-determinable condition is satisfied
    """
    unique_set = [v[0] for k, v in same_color_partition.items() if len(v) == 1]
    if len(unique_set) == len(same_color_partition.keys()):
        return True
    else:
        return False



def find_group(node, partition):
    for group, nodes in partition.items():
        if node in nodes:
            return group  # Return the key of the group
    return f'{node} not found in {partition}!'  # Return None if node is not found

def get_nbhd(node_index_list, unique_set, Adj):
    """Get the neighbors of a list of nodes"""
    nbhd = []
    # For each node
    for i in node_index_list:
        # Get all neighbors of the node (positions in Adj[i] with value 1)
        neighbors = np.where(Adj[i] > 0)[0]
        for j in neighbors:
            if j not in unique_set:
                nbhd.append(int(j))
    return nbhd

def get_cluster(same_color_partition,unique_set,partition_index,sub_graph_nodes,Adj):
    labeled_nodes = list(chain.from_iterable(sub_graph_nodes.values()))
    for i in same_color_partition[partition_index]:
        if i not in labeled_nodes:
            sub_graph_i = []
            sub_graph_i.append(i)
            sub_graph_nodes[i] = list(set(sub_graph_i))
        else:
            continue
    previous_sum = 0
    
    while sum([len(v) for k,v in sub_graph_nodes.items() if k in same_color_partition[partition_index]]) -previous_sum >0:
        previous_sum = sum([len(v) for k,v in sub_graph_nodes.items()if k in same_color_partition[partition_index]])
        for i in same_color_partition[partition_index]:
            if i not in labeled_nodes:
                sub_graph_nodes[i] = sub_graph_nodes[i]+get_nbhd(sub_graph_nodes[i],unique_set,Adj)
                sub_graph_nodes[i] = set(sub_graph_nodes[i])
                sub_graph_nodes[i] = list(set(sub_graph_nodes[i]))
    return sub_graph_nodes

def derive_subgraphs(same_color_partition,Adj):
    """
    Divide subgraphs based on node colors
    """
    unique_set = [v[0] for k, v in same_color_partition.items() if len(v) == 1]
    same_color_partition = {k: v for k, v in same_color_partition.items() if len(v) > 1}
    sub_graph_nodes = {}
    multiplicity_nodes = [item for item in list(range(len(Adj))) if item not in unique_set]
    condition = True
    state = 'uncheck'
    while condition:
        labeled_nodes = list(chain.from_iterable(sub_graph_nodes.values()))
        for i in multiplicity_nodes:
            if i not in labeled_nodes:
                partition_index = find_group(i,same_color_partition)
                sub_graph_nodes = get_cluster(same_color_partition,unique_set,partition_index,sub_graph_nodes,Adj)
        if sum([len(v) for v in sub_graph_nodes.values()]) < len(Adj)-len(unique_set):
            condition = True
            state = 'continue'
        elif sum([len(v) for v in sub_graph_nodes.values()]) == len(Adj) - len(unique_set):
            condition = False
            state = 'complete'
        else:
            condition = False
            state = 'error'
        
    for i in list(sub_graph_nodes.keys()):
        sub_graph_nodes[i] = set(sub_graph_nodes[i])
    return sub_graph_nodes,state


def check_distinct(sub_graph_nodes,stable_partition):
    for k,v in sub_graph_nodes.items():
        for p,q in stable_partition.items():
            if len(v.intersection(q)) > 1:
                print(f'{v}intersection{q} greater than 1')
                return False
    return True


def check_graph_disjoint(sub_graph_nodes):
    for i in sub_graph_nodes:
        for j in sub_graph_nodes:
            if i != j:
                if len(sub_graph_nodes[i].intersection(sub_graph_nodes[j])) != 0:
                    return False
    return True

def connectivity_check(dict1,dict2,Adj):
    for i in dict1:
        for j in dict2:
            if Adj[i][j] != 0:
                return True
    return False

def check_graph_disconnect(sub_graph_nodes,Adj):
    for i in sub_graph_nodes:
        for j in sub_graph_nodes:
            if i != j:
                if connectivity_check(sub_graph_nodes[i],sub_graph_nodes[j],Adj):
                    return False
    return True

def check_symmetric_decomposable(same_color_partition,Adj):
    """
    Check if the graph is symmetric decomposable
    """
    sub_graph_nodes,state = derive_subgraphs(same_color_partition,Adj)
    same_color_partition = {k: v for k, v in same_color_partition.items() if len(v) > 1}

    if state == 'error':
        #print('False: Graph is not symmetric decomposable due to error.')
        return False, "Due to decompose error"
    if not check_distinct(sub_graph_nodes,same_color_partition):
        return False, "Due to distinctness."
    elif check_graph_disjoint(sub_graph_nodes):
        if check_graph_disconnect(sub_graph_nodes,Adj):
            #print('True: Graph is symmetric decomposable.')
            return True, 'Symmetric Decomposable.'
        else:
            #print('False: Graph is not symmetric decomposable due to connectivity.')
            return False, "Due to connectivity."
    else:
        #print('False: Graph is not symmetric decomposable due to disjointness.')
        return False, "Due to disjointness."

def check_sufficient_conditions(same_color_partition,Adj):
    """
    Check if the sufficient conditions are satisfied
    """
    if check_wl_determinable(same_color_partition):
        sufficient_cond = True
        msg = "WL-determinable"
    else:
        check_sym_decom,msg = check_symmetric_decomposable(same_color_partition,Adj)
        if check_sym_decom:
            sufficient_cond = True
        else:
            sufficient_cond = False
    return sufficient_cond, msg

# Evaluate whether two samples satisfy the sufficient conditions
def check_sufficiency_two(same_color_partition1,Adj1, same_color_partition2,Adj2):
    check_sufficient1,msg1 = check_sufficient_conditions(same_color_partition1,Adj1)
    check_sufficient2,msg2 = check_sufficient_conditions(same_color_partition2,Adj2)
    msg = {'msg1':msg1, 'msg2':msg2}
    if check_sufficient1 and check_sufficient2:
        sufficient = True
    else:
        sufficient = False
    return sufficient, msg

def transform_constriant_with_0_rhs_ineq(model):
    all_cons = model.getConstrs()

    for cons in all_cons:
        if cons.RHS == 0 and cons.Sense == GRB.LESS_EQUAL:
            # Change to >=
            cons.setAttr("Sense", GRB.GREATER_EQUAL)
            cons.setAttr("RHS", 0.0)

            row = model.getRow(cons)   # This is available in Gurobi 9!
            rhs = cons.RHS

            # Remove the old constraint
            model.remove(cons)

            # Construct the new constraint
            new_expr = gp.LinExpr()
            for i in range(row.size()):
                v = row.getVar(i)
                c = row.getCoeff(i)
                new_expr.addTerms(-c, v)

            new_cons = model.addConstr(new_expr >= -rhs)
            # Update the model after the loop
            model.update()
        else:
            continue
    return model

def transform_constriant_with_0_rhs_eq(model, save_path):
    """
    Standardize equality constraints with rhs=0 in Gurobi model.
    
    Rules:
    1. For each equality constraint with rhs=0:
       - Extract the variables and coefficients
       - If the sum of coefficients > 0: keep as is
       - If the sum of coefficients < 0: negate all coefficients
    2. If the sum of coefficients == 0, save the model to save_path (if provided).

    Parameters
    ----------
    model : gurobipy.Model
        Input model
    save_path : str
        Path to save the model if a special case (sum(coeffs)=0) occurs (.lp or .mps)
    
    Return
    ----
    model : gurobipy.Model
        Updated model (in-place modification)
    """
    needs_save = False

    for constr in model.getConstrs():
        # Only process equality constraints with rhs=0
        if constr.Sense == '=' and abs(constr.RHS) < 1e-9:
            expr = model.getRow(constr)
            coeffs = [expr.getCoeff(i) for i in range(expr.size())]
            vars_ = [expr.getVar(i) for i in range(expr.size())]

            coeff_sum = sum(coeffs)

            if abs(coeff_sum) < 1e-12:
                needs_save = True
                continue  # Do not modify, but mark for saving
            
            # If the sum < 0, negate all coefficients
            if coeff_sum < 0:
                coeffs = [-c for c in coeffs]

            # Remove the old constraint, add the new constraint
            model.remove(constr)
            model.addConstr(sum(c * v for c, v in zip(coeffs, vars_)) == 0)

    model.update()
    #model.write("transformed_model_eq.lp")
    if needs_save and save_path is not None:
        model.write(save_path)

    return model


def check_lp_equivalence(lp_path1, lp_path2, verbose: bool = False):
    """
    Check whether two LP files are equivalent.

    Args:
        lp_path1: Path to the first LP file
        lp_path2: Path to the second LP file
        verbose: Whether to print detailed information
    Returns:
        tuple: (Whether equivalent, Error information)
    """

    info = {}
    time_info = {}

    with gp.Env(empty=True) as env:
        env = gp.Env(empty=True)
        env.setParam("LogToConsole", 0)
        env.start()
        model1 = gp.read(lp_path1,env)
        try:
            model2 = gp.read(lp_path2,env)
        except Exception as e:
            return False, f"Error reading LP file: {e}",time_info
        
        # Standardize boundary constraints
        model1 = convert_cons2boundary(model1)
        model1 = transform_constriant_with_0_rhs_ineq(model1)
        model1 = transform_constriant_with_0_rhs_eq(model1,'failure/'+lp_path1.split('/')[-1]+'failure.lp')
        model1.write('model_ref.lp')
        model2 = convert_cons2boundary(model2)
        model2 = transform_constriant_with_0_rhs_ineq(model2)
        model2 = transform_constriant_with_0_rhs_eq(model2,'failure/'+lp_path2.split('/')[-1]+'failure.lp')
        model2.write('model_test.lp')

        # Check basic properties
        if model1.NumVars != model2.NumVars:
            info['var_num_check'] = False
            return False, info, time_info
        else:
            info['var_num_check'] = True
            
        if model1.NumConstrs != model2.NumConstrs:
            info['cons_num_check'] = False
            return False, info, time_info
        else:
            info['cons_num_check'] = True

        # Generate graphs and perform WL-Test
        start_time = time.time()
        G1 = generate_bipartite(model1)
        partition1, iter1 = wl_test_nx(G1)        
        G2 = generate_bipartite(model2)
        partition2, iter2 = wl_test_nx(G2)
        end_time = time.time()
        wl_coloring_time = end_time - start_time
        time_info['wl_coloring_time'] = wl_coloring_time
        if not (partition1.keys() == partition2.keys()):
            info['wl_check'] = False
            return False, info, time_info
        else:
            # Check whether the number of nodes in each partition is consistent
            #start_time = time.time()
            #for key in partition1.keys():
            #    if len(partition1[key]) != len(partition2[key]):
            #        info['color match'] = False
            #        return False, info
            #    else:
            #        continue
            #end_time = time.time()
            #coloring_mathing_time = end_time - start_time
            #time_info['coloring_mathing_time'] = coloring_mathing_time
            # Check whether the sufficient condition is met
            start_time = time.time()
            Adj1 = nx.adjacency_matrix(G1).toarray()
            Adj2 = nx.adjacency_matrix(G2).toarray()
            check_sufficiency, sufficient_msg = check_sufficiency_two(partition1, Adj1,partition2,Adj2)
            end_time = time.time()
            sufficient_check_time = end_time - start_time
            time_info['sufficient_check_time'] = sufficient_check_time
            if check_sufficiency:
                info['sufficient_check'] = True
                info['sufficient_msg'] = sufficient_msg
                return True, info, time_info
            else:
                info['sufficient_check'] = False
                info['sufficient_msg'] = sufficient_msg

                return False, info, time_info
                
                

        # Only test compilation
        env.close()
        return True, "No other check", time_info

