import time
import numpy as np
import torch as th
import argparse
import json
import utils
import pickle
from model import LinkPredict

def main(args, dir_path, device, save_score, metric):
    T_start = time.time();
    print("Start Review", dir_path);
    
    with open(dir_path + "gene_to_idx.pkl", "rb") as f: gene_to_idx = pickle.load(f);
    _, eval_cell_idx, num_nodes, _, \
    train_edge_idx, train_node_idx, eval_node_idx, eval_edge_list, eval_edge_ulist = \
        utils.load_data(args.cells, gene_to_idx);
    num_rels = len(args.cells);
    
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
    
    link_predict = LinkPredict(
        train_edge_idx,
        train_node_idx,
        enc_param,
        dec_param,
        None,
        None,
        device,
        "{}model{}.pth".format(dir_path, metric)
    );
    
    hits = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500, 1000];
    eval_result = link_predict.evaluate(eval_edge_list,
                                        eval_edge_ulist,
                                        eval_node_idx,
                                        eval_cell_idx,
                                        10000,
                                        hits,
                                        True,
                                        save_score);

    with open(dir_path + "review" + metric + ".pkl", "wb") as f: pickle.dump(eval_result, f);
    print("End. took {}".format(utils.hms(time.time() - T_start)));

if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    
    parser.add_argument('--dir', default = "");
    parser.add_argument('--result');
    parser.add_argument('--gpu', default = 0);
    parser.add_argument('--score', default = 0, type = int); # if 1, stores entire scores
    parser.add_argument('--metric', default = "");
    pre_args = parser.parse_args();
    
    dir_path = "Results/{}/Result{}/".format(pre_args.dir, pre_args.result) if pre_args.dir != "" \
          else "Results/Result{}/".format(pre_args.result);
    
    with open(dir_path + "hparam.json", "r") as f:
        args = argparse.Namespace(**json.load(f));
    
    main(args, dir_path, th.device("cuda:" + str(pre_args.gpu)), pre_args.score, pre_args.metric);
    