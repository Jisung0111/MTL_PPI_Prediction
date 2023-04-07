import time
import numpy as np
import torch as th
import argparse
import json
import utils
import pickle
from model import LinkPredict

def main(args):
    T_start = time.time();
    device = th.device(args.gpu);
    utils.seed_init(args.seed, device);
    
    eval_cells, eval_cell_idx, num_nodes, gene_to_idx,\
    train_edge_idx, train_node_idx, eval_node_idx, eval_edge_list, eval_edge_ulist = \
        utils.load_data(args.cells);
    
    eval_cell_names = eval_cells + ["ALL"];
    target_cell_idx = eval_cell_names.index(args.target_cell);
    
    num_rels = len(args.cells);
    
    args_to_send = vars(args);
    args_to_send["cells"] = args.cells;
    args_to_send["eval_cells"] = eval_cells;
    result_path = utils.make_result_dir(",".join(args.cells));
    utils.log("start", [gene_to_idx, args_to_send, T_start], result_path);
    
    enc_param = {
        "num_nodes" : num_nodes,
        "num_rels"  : num_rels,
        "num_layers": args.num_layers,
        "head"      : args.head,
        "in_dim"    : args.in_dim,
        "out_dim"   : args.out_dim,
        "dropout"   : args.dropout,
    };
    
    dec_param = {
        "decoder"     : args.decoder,
        "num_channels": args.num_channels,
        "kernel_size" : args.kernel_size
    };
    
    opt_param = {
        "lr"       : args.lr,
        "l2_reg"   : args.l2_reg,
        "scheduler": args.scheduler,
        # CosineAnnealingWarmRestarts
        "T_0"      : args.T_0,
        "T_mult"   : args.T_mult,
        "etamin"   : args.etamin,
        # ReduceLROnPlateau
        "factor"   : args.factor,
        "patience" : args.patience,
        "min_lr"   : args.min_lr
    };
    
    lrn_param = {
        "lbl_smooth"     : args.lbl_smooth,
        "batch_size"     : args.batch_size,
        "grad_norm"      : args.grad_norm
    }
    
    link_predict = LinkPredict(
        train_edge_idx,
        train_node_idx,
        enc_param,
        dec_param,
        opt_param,
        lrn_param,
        device
    );
    
    history = {
        "Hit@K": [],
        "MRR"  : [],
        "MR"   : [],
        "AUROC": [],
        "Epoch": [],
        "Loss" : []
    };
    
    epoch, early_stop = 0, 100;
    best_mrr_idx, best_mrr_epoch, best_mrr = 0, 0, 0;
    best_mr_idx, best_mr_epoch, best_mr = 0, 0, 123456789;
    best_auc_idx, best_auc_epoch, best_auc = 0, 0, 0;
    T_eval = time.time();
    eval_result = link_predict.evaluate(eval_edge_list, eval_edge_ulist, eval_node_idx, eval_cell_idx);
    utils.log("eval", [eval_result, T_eval, target_cell_idx], result_path);
    
    history["Epoch"].append(epoch);
    for metric in eval_result: history[metric].append(eval_result[metric]);
    
    while epoch < args.epoch:
        T_train = time.time();
        with th.cuda.device(device): th.cuda.empty_cache();
        
        epoch, loss = link_predict.run_epoch(epoch);
        history["Loss"].append(loss);
        utils.log("train", [epoch, loss, T_start, T_train], result_path);
        
        if epoch % args.eval_period == 0:
            T_eval = time.time();
            with th.cuda.device(device): th.cuda.empty_cache();
            
            eval_result = link_predict.evaluate(eval_edge_list, eval_edge_ulist, eval_node_idx, eval_cell_idx);
            utils.log("eval", [eval_result, T_eval, target_cell_idx], result_path);
            
            history["Epoch"].append(epoch);
            for metric in eval_result: history[metric].append(eval_result[metric]);
            
            with open(result_path + "history.pkl", "wb") as f: pickle.dump(history, f);
            
            if best_mrr <= eval_result["MRR"][1][target_cell_idx]: # valid set, target cell
                link_predict.save(result_path + "modelMRR.pth");
                best_mrr, best_mrr_idx = eval_result["MRR"][1][target_cell_idx], len(history["Epoch"]) - 1;
                best_mrr_epoch = epoch;
            
            if best_mr >= eval_result["MR"][1][target_cell_idx]: # valid set, target cell
                link_predict.save(result_path + "modelMR.pth");
                best_mr, best_mr_idx = eval_result["MR"][1][target_cell_idx], len(history["Epoch"]) - 1;
                best_mr_epoch = epoch;
            
            if best_auc <= eval_result["AUROC"][1][target_cell_idx]: # valid set, target cell
                link_predict.save(result_path + "modelAUROC.pth");
                best_auc, best_auc_idx = eval_result["AUROC"][1][target_cell_idx], len(history["Epoch"]) - 1;
                best_auc_epoch = epoch;
            
            if epoch - best_mrr_epoch > early_stop and \
               epoch - best_mr_epoch  > early_stop and \
               epoch - best_auc_epoch > early_stop: break;
    
    utils.log("finish", [best_mrr_idx, history, T_start, "MRR", True, target_cell_idx], result_path);
    utils.log("finish", [best_mr_idx, history, T_start, "MR", False, target_cell_idx], result_path);
    utils.log("finish", [best_auc_idx, history, T_start, "AUROC", False, target_cell_idx], result_path);

if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    
    parser.add_argument('--hparam');
    file_loc = parser.parse_args();
    
    default_hparam = {
        "gpu"            : 0,           # default is cuda:0
        "seed"           : 1,
        "epoch"          : 200,
        "cells"          : [],          # idx0 cell is target cell
        "target_cell"    : "ALL",
        "batch_size"     : 512,
        "num_layers"     : 1,
        "head"           : 2,
        "in_dim"         : 100,
        "out_dim"        : 200,
        "decoder"        : "ConvE",
        "num_channels"   : 192,
        "kernel_size"    : 7,
        "lr"             : 0.001,
        "l2_reg"         : 0.0,
        "scheduler"      : "None",
        "T_0"            : 20,
        "T_mult"         : 1,
        "etamin"         : 1e-6,
        "factor"         : 0.1,
        "patience"       : 5,
        "min_lr"         : 1e-6,
        "lbl_smooth"     : 0.1,
        "dropout"        : 0.3,
        "grad_norm"      : 1.0,
        "eval_period"    : 1            # evaluation period (unit: epoch)
    }
    
    with open('Exps/' + file_loc.hparam) as json_file:
        default_hparam.update(json.load(json_file));
        args = argparse.Namespace(**default_hparam);
    
    main(args);
    