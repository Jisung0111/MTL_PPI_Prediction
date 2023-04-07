import numpy as np
import torch as th
import random
import os, json
import pickle
import time

def seed_init(seed, device):
    th.manual_seed(seed);
    if device != th.device("cpu"):
        with th.cuda.device(device): th.cuda.manual_seed(seed);
    np.random.seed(seed);
    random.seed(seed);
    th.backends.cudnn.deterministic = True;
    th.backends.cudnn.benchmark = False;
    
def hms(x):
    y = int(x);
    return "{}h {:02d}m {:02d}.{:01d}s".format(y // 3600, y // 60 % 60, y % 60, int(x % 1 * 10));

def make_result_dir(result_dir):
    result_path = "Results/" + result_dir + "/";
    if result_dir not in os.listdir("Results"): os.mkdir(result_path);
    result_dirs = os.listdir(result_path);
    result_idx = 0;
    while "Result{}".format(result_idx) in result_dirs: result_idx += 1;
    result_path = "{}/Result{}".format(result_path, result_idx);
    os.mkdir(result_path);
    return result_path + '/';

def load_data(cells, gene_to_idx = {}):
    '''
    input
        cells: list(#rel)
        ebatch_size: int
        ulist: bool
    
    output
        eval_cells: list
        eval_cell_idx: list
        num_nodes: int
        gene_to_idx: dict{gene: idx}
        train_edge_idx: numpy(3(s, r, o), E)
        train_node_idx: list(#rel, numpy(#node_in_rel))
        eval_node_idx: list(#rel, numpy(#node_in_rel))
        eval_edge_list: list(3(split), #rel, numpy([[s', ...], [o, ...]])) # every edges
        option(if u_list is True)
        eval_edge_ulist: list(3(split), #rel, numpy([[s', ...], [o, ...]])) # upper triangular edges
    '''
    data_dir = "Data";
    merged_case = ' ' in cells[0];
    data_dir_list = os.listdir(data_dir);
    eval_cells = [cell for cell in " ".join(cells).split(" ") if cell + "_test.txt" in data_dir_list];
    eval_cell_idx = [0 if merged_case else i for i in range(len(eval_cells))];
    
    if len(gene_to_idx): num_nodes = len(gene_to_idx);
    else:
        for r, cell in enumerate(cells):
            with open("{}/{}.txt".format(data_dir, cell), 'r') as f:
                train_data = [k[:-1].split("\t") for k in f.readlines()];
                for s, o in train_data:
                    gene_to_idx[s] = gene_to_idx.get(s, len(gene_to_idx));
                    gene_to_idx[o] = gene_to_idx.get(o, len(gene_to_idx));
        
        num_nodes = len(gene_to_idx);
        rand_idx = np.random.permutation(num_nodes);
        for gene in gene_to_idx: gene_to_idx[gene] = rand_idx[gene_to_idx[gene]];

    if merged_case: train_edge_set = set();
    # train_edge_set = set();
    
    train_edge_idx = set();
    train_node_idx = [set() for r in range(len(cells))];
    for r, cell in enumerate(cells):
        with open("{}/{}.txt".format(data_dir, cell), 'r') as f:
            train_data = [k[:-1].split("\t") for k in f.readlines()];
            for s, o in train_data:
                si, oi = gene_to_idx[s], gene_to_idx[o];
                train_edge_idx.add((si, r, oi)); train_node_idx[r].add(si);
                train_edge_idx.add((oi, r, si)); train_node_idx[r].add(oi);
                if merged_case:
                    train_edge_set.add((si, oi));
                    train_edge_set.add((oi, si));
                # train_edge_set.add((si, oi));
                # train_edge_set.add((oi, si));
    train_edge_idx = np.array(list(train_edge_idx)).T;
    train_node_idx = [np.array(sorted(list(train_node_idx[r]))) for r in range(len(cells))];
    
    eval_edge_idx = [set() for split in range(3)];
    eval_node_idx = [set() for r in range(len(eval_cells))];
    for split in range(3):
        for r, cell in enumerate(eval_cells):
            with open("{}/{}{}.txt".format(data_dir, cell, ["", "_valid", "_test"][split]), 'r') as f:
                for s, o in [k[:-1].split("\t") for k in f.readlines()]:
                    si, oi = gene_to_idx[s], gene_to_idx[o];
                    nsp = 0 if merged_case and ((si, oi) in train_edge_set) else split;
                    # nsp = 0 if (si, oi) in train_edge_set else split;
                    eval_edge_idx[nsp].add((si, r, oi)); eval_node_idx[r].add(si);
                    eval_edge_idx[nsp].add((oi, r, si)); eval_node_idx[r].add(oi);
    eval_node_idx = [np.array(sorted(list(eval_node_idx[r]))) for r in range(len(eval_cells))];
    node_rev_idx = [{i: j for j, i in enumerate(eval_node_idx[r])} for r in range(len(eval_cells))];
    
    eval_edge_list = [[[] for rel in range(len(eval_cells))] for split in range(3)];
    for split in range(3):
        for s, r, o in eval_edge_idx[split]:
            eval_edge_list[split][r].append([node_rev_idx[r][s], node_rev_idx[r][o]]);
        
        for rel in range(len(eval_cells)):
            eval_edge_list[split][rel] = np.array(eval_edge_list[split][rel]).T;
    
    eval_edge_ulist = [[[] for rel in range(len(eval_cells))] for split in range(3)];
    for split in range(3):
        for s, r, o in eval_edge_idx[split]:
            ns, no = node_rev_idx[r][s], node_rev_idx[r][o];
            if ns <= no: eval_edge_ulist[split][r].append([ns, no]);
        
        for rel in range(len(eval_cells)):
            eval_edge_ulist[split][rel] = np.array(eval_edge_ulist[split][rel]).T;
        
    return eval_cells, eval_cell_idx, num_nodes, gene_to_idx, train_edge_idx, train_node_idx, eval_node_idx, eval_edge_list, eval_edge_ulist;

def log(phase, args, result_path):
    if phase == "start":
        gene_to_idx, hparam, T_start = args;
        with open(result_path + "hparam.json", 'w') as f: json.dump(hparam, f, indent = 4);
        with open(result_path + "gene_to_idx.pkl", 'wb') as f: pickle.dump(gene_to_idx, f);
        with open(result_path + "train_log.txt", 'w') as f:
            f.write("Training Start ({})\n".format(hms(time.time() - T_start)));
    
    elif phase == "train":
        epoch, loss, T_start, T_train = args;
        with open(result_path + "train_log.txt", 'a') as f:
            f.write("Epoch {} ({})\tLoss: {:.5f}\ttook {:.2f}s\n".format(epoch, hms(time.time() - T_start), loss, time.time() - T_train));
    
    elif phase == "eval":
        eval_result, T_eval, idx = args;
        with open(result_path + "train_log.txt", 'a') as f:
            f.write("\tTrain set,  MRR: {:.4f}\tMR: {:.1f}\tAUROC: {:.4f}\tHit@1: {:.4f}\tHit@10: {:.4f}\tHit@100: {:.4f}\n"
                    "\tValid set,  MRR: {:.4f}\tMR: {:.1f}\tAUROC: {:.4f}\tHit@1: {:.4f}\tHit@10: {:.4f}\tHit@100: {:.4f}\n"
                    "\tTest  set,  MRR: {:.4f}\tMR: {:.1f}\tAUROC: {:.4f}\tHit@1: {:.4f}\tHit@10: {:.4f}\tHit@100: {:.4f}\n"
                    "\ttook {:.2f}s\n".format(
                eval_result["MRR"][0][idx], eval_result["MR"][0][idx], eval_result["AUROC"][0][idx], eval_result["Hit@K"][0][idx][0], eval_result["Hit@K"][0][idx][1], eval_result["Hit@K"][0][idx][2],
                eval_result["MRR"][1][idx], eval_result["MR"][1][idx], eval_result["AUROC"][1][idx], eval_result["Hit@K"][1][idx][0], eval_result["Hit@K"][1][idx][1], eval_result["Hit@K"][1][idx][2],
                eval_result["MRR"][2][idx], eval_result["MR"][2][idx], eval_result["AUROC"][2][idx], eval_result["Hit@K"][2][idx][0], eval_result["Hit@K"][2][idx][1], eval_result["Hit@K"][2][idx][2],
                time.time() - T_eval)
            );
    
    elif phase == "finish":
        best_idx, history, T_start, criteria, write_done, idx = args;
        eval_result = {metric: history[metric][best_idx] for metric in ["MRR", "MR", "Hit@K", "AUROC"]};
        with open(result_path + "history.pkl", 'wb') as f: pickle.dump(history, f);
        
        with open(result_path + "train_log.txt", 'a') as f:
            if write_done: f.write("\nTraining Done ({})".format(hms(time.time() - T_start)));
            f.write("\n\nBest {} on Epoch {}\n"
                    "\tTrain set,  MRR: {:.4f}\tMR: {:.1f}\tAUROC: {:.4f}\tHit@1: {:.4f}\tHit@10: {:.4f}\tHit@100: {:.4f}\n"
                    "\tValid set,  MRR: {:.4f}\tMR: {:.1f}\tAUROC: {:.4f}\tHit@1: {:.4f}\tHit@10: {:.4f}\tHit@100: {:.4f}\n"
                    "\tTest  set,  MRR: {:.4f}\tMR: {:.1f}\tAUROC: {:.4f}\tHit@1: {:.4f}\tHit@10: {:.4f}\tHit@100: {:.4f}".format(
                criteria, history["Epoch"][best_idx],
                eval_result["MRR"][0][idx], eval_result["MR"][0][idx], eval_result["AUROC"][0][idx], eval_result["Hit@K"][0][idx][0], eval_result["Hit@K"][0][idx][1], eval_result["Hit@K"][0][idx][2],
                eval_result["MRR"][1][idx], eval_result["MR"][1][idx], eval_result["AUROC"][1][idx], eval_result["Hit@K"][1][idx][0], eval_result["Hit@K"][1][idx][1], eval_result["Hit@K"][1][idx][2],
                eval_result["MRR"][2][idx], eval_result["MR"][2][idx], eval_result["AUROC"][2][idx], eval_result["Hit@K"][2][idx][0], eval_result["Hit@K"][2][idx][1], eval_result["Hit@K"][2][idx][2])
            );

def save_json(hparam, write_on_sh = True):
    cnt, dir_list = 0, os.listdir('Exps');
    while 'Exp' + str(cnt) + '.json' in dir_list: cnt += 1;
    file_name = 'Exps/Exp' + str(cnt) + '.json';
    
    if write_on_sh:
        gpu = hparam.get("gpu", 0);
        with open("Train{}.sh".format(gpu), "a") as f:
            f.write('python train.py --hparam ' + file_name.split('/')[-1] + '\n');

    with open(file_name, 'w') as f: json.dump(hparam, f, indent = 4);
    return file_name;

def best_find(val_hstry, tst_hstry, epochs, lrg_good):
    idx = np.argmax(val_hstry) if lrg_good else np.argmin(val_hstry);
    return tst_hstry[idx], epochs[idx];
        