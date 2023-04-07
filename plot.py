import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import utils

parser = argparse.ArgumentParser();
    
parser.add_argument('--dir', default = "");
parser.add_argument('--result');
parser.add_argument('--opt', default = 't'); # combination of t, a, e (target, all, every)
parser.add_argument('--metric', default = "");
pre_args = parser.parse_args();

if pre_args.metric not in ["MRR", "MR", "AUROC"]:
    raise ValueError("Wrong metric name");

dir_path = "Results/{}/Result{}/".format(pre_args.dir, pre_args.result) if pre_args.dir != "" \
      else "Results/Result{}/".format(pre_args.result);

with open(dir_path + "hparam.json", "r") as f:
    args = argparse.Namespace(**json.load(f));

cells, cell_names = [], args.eval_cells + ["ALL"];
num_rels = len(args.eval_cells);

eval_cell_names = args.eval_cells + ["ALL"];

if 't' in pre_args.opt: cells.append(eval_cell_names.index(args.target_cell));
if 'e' in pre_args.opt: cells = [i for i in range(num_rels + 1)];

with open(dir_path + "history.pkl", "rb") as f: history = pickle.load(f);
if "review" + pre_args.metric + ".pkl" in os.listdir(dir_path):
    with open(dir_path + "review" + pre_args.metric + ".pkl", "rb") as f: review = pickle.load(f);
else: review = None;

if "Loss" in history:
    plt.figure(figsize = (8, 5));
    plt.title("Loss");
    plt.yscale('logit');
    plt.plot(np.arange(len(history["Loss"])) + 1, history["Loss"]);
    plt.savefig(dir_path + "Loss.jpg", dpi = 200);

N = len(history["Epoch"]);
tolist = lambda m, x, y: [history[m][i][x][y] for i in range(N)];
tolist2 = lambda m, x, y, z: [history[m][i][x][y][z] for i in range(N)];
cuml_to_dnst = lambda arr: [arr[i + 1] - arr[i] for i in range(len(arr) - 1)];

HITS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 200, 500, 1000];

for cell in cells:
    plt.figure(figsize = (36, 24));
    
    plt.subplot(4, 4, 1);
    best_val, best_epoch = utils.best_find(tolist("AUROC", 1, cell), tolist("AUROC", 2, cell), history["Epoch"], True);
    plt.title("AUROC (Best: {:.4f} on Epoch {})".format(best_val, best_epoch));
    plt.ylabel("ROC-AUC");
    plt.yscale("linear");
    plt.xlabel("Epoch");
    
    plt.plot(history["Epoch"], tolist("AUROC", 0, cell), label = "Train", color = [0.8, 0.2, 0.2, 1.0]);
    plt.plot(history["Epoch"], tolist("AUROC", 1, cell), label = "Valid", color = [0.2, 0.8, 0.2, 1.0]);
    plt.plot(history["Epoch"], tolist("AUROC", 2, cell), label = "Test", color = [0.2, 0.2, 0.8, 1.0]);
    plt.legend();
    
    plt.subplot(4, 4, 2);
    best_val, best_epoch = utils.best_find(tolist("MRR", 1, cell), tolist("MRR", 2, cell), history["Epoch"], True);
    plt.title("MRR (Best: {:.4f} on Epoch {})".format(best_val, best_epoch));
    plt.ylabel("Mean Reciprocal Rank");
    plt.yscale("linear");
    plt.xlabel("Epoch");
    
    plt.plot(history["Epoch"], tolist("MRR", 0, cell), label = "Train", color = [0.8, 0.2, 0.2, 1.0]);
    plt.plot(history["Epoch"], tolist("MRR", 1, cell), label = "Valid", color = [0.2, 0.8, 0.2, 1.0]);
    plt.plot(history["Epoch"], tolist("MRR", 2, cell), label = "Test", color = [0.2, 0.2, 0.8, 1.0]);
    plt.legend();
    
    plt.subplot(4, 4, 3);
    best_val, best_epoch = utils.best_find(tolist("MR", 1, cell), tolist("MR", 2, cell), history["Epoch"], False);
    plt.title("MR (Best: {:.1f} on Epoch {})".format(best_val, best_epoch));
    plt.ylabel("Mean Rank");
    plt.yscale("log");
    plt.xlabel("Epoch");
    
    plt.plot(history["Epoch"], tolist("MR", 0, cell), label = "Train", color = [0.8, 0.2, 0.2, 1.0]);
    plt.plot(history["Epoch"], tolist("MR", 1, cell), label = "Valid", color = [0.2, 0.8, 0.2, 1.0]);
    plt.plot(history["Epoch"], tolist("MR", 2, cell), label = "Test", color = [0.2, 0.2, 0.8, 1.0]);
    plt.legend();
    
    for hit_idx, hit in enumerate([1, 10, 100]):
        plt.subplot(4, 4, 5 + hit_idx);
        best_val, best_epoch = utils.best_find(tolist2("Hit@K", 1, cell, hit_idx), tolist2("Hit@K", 2, cell, hit_idx), history["Epoch"], True);
        plt.title("Hit @ {} (Best: {:.4f} on Epoch {})".format(hit, best_val, best_epoch));
        plt.ylabel("Hit @ {}".format(hit));
        plt.yscale("linear");
        plt.xlabel("Epoch");
        
        plt.plot(history["Epoch"], tolist2("Hit@K", 0, cell, hit_idx), label = "Train", color = [0.8, 0.2, 0.2, 1.0]);
        plt.plot(history["Epoch"], tolist2("Hit@K", 1, cell, hit_idx), label = "Valid", color = [0.2, 0.8, 0.2, 1.0]);
        plt.plot(history["Epoch"], tolist2("Hit@K", 2, cell, hit_idx), label = "Test", color = [0.2, 0.2, 0.8, 1.0]);
        plt.legend();
    
    if review is not None:
        x_axis = np.linspace(0.0, 1.0, 101);
        x_axis = (x_axis[1:] + x_axis[:-1]) / 2;
        
        plt.subplot(8, 8, 7);
        plt.title("Score Distribution");
        plt.ylabel("# Edges");
        plt.yscale("log");
        # plt.yscale("linear");
        
        plt.plot(x_axis, cuml_to_dnst(review["Score_Dist"][0][cell]), label = "Train", color = [0.8, 0.2, 0.2, 1.0]);
        plt.legend();
        
        plt.subplot(8, 8, 8);
        plt.yscale("log");
        # plt.yscale("linear");
        plt.plot(x_axis, cuml_to_dnst(review["Score_Dist"][1][cell]), label = "Valid", color = [0.2, 0.8, 0.2, 1.0]);
        plt.legend();
        
        plt.subplot(8, 8, 15);
        plt.ylabel("# Edges");
        plt.yscale("log");
        # plt.yscale("linear");
        plt.xlabel("Score");
        plt.plot(x_axis, cuml_to_dnst(review["Score_Dist"][2][cell]), label = "Test", color = [0.2, 0.2, 0.8, 1.0]);
        plt.legend();
        
        plt.subplot(8, 8, 16);
        plt.yscale("log");
        # plt.yscale("linear");
        plt.xlabel("Score");
        plt.plot(x_axis, cuml_to_dnst(review["Score_Dist"][3][cell]), label = "Rest", color = [0.8, 0.8, 0.2, 1.0]);
        plt.legend();
        
        if cell != num_rels:
            metric = "Accuracy";
            for metric_idx, metric in enumerate(["Accuracy", "Sensitivity", "Precision"]):
                plt.subplot(4, 4, 8 + metric_idx);
                plt.title("Sensitivity (a.k.a. TPR or Recall), Specificity" if metric == "Sensitivity" else metric);
                plt.ylabel("TPR (Recall), Specificity" if metric == "Sensitivity" else metric);
                plt.yscale("linear");
                plt.xlabel("Threshold");
                
                plt.plot(review["Threshold"][0][cell], review[metric][0][cell], label = "Train", color = [0.8, 0.2, 0.2, 1.0]);
                plt.scatter(review["Threshold"][0][cell], review[metric][0][cell], edgecolors = [0.8, 0.2, 0.2, 1.0], color = 'w', s = 10);
                plt.plot(review["Threshold"][1][cell], review[metric][1][cell], label = "Valid", color = [0.2, 0.8, 0.2, 1.0]);
                plt.scatter(review["Threshold"][1][cell], review[metric][1][cell], edgecolors = [0.2, 0.8, 0.2, 1.0], color = 'w', s = 10);
                plt.plot(review["Threshold"][2][cell], review[metric][2][cell], label = "Test", color = [0.2, 0.2, 0.8, 1.0]);
                plt.scatter(review["Threshold"][2][cell], review[metric][2][cell], edgecolors = [0.2, 0.2, 0.8, 1.0], color = 'w', s = 10);
                if metric == "Sensitivity":
                    plt.plot(review["Threshold"][2][cell], review["Specificity"][2][cell], label = "Specificity", color = [0.3, 0.3, 0.3, 1.0]);
                plt.legend();
            
            for metric_idx, metric in enumerate(["MCC", "F1_Score"]):
                metnam = metric.replace("_", " ");
                plt.subplot(4, 4, 11 + metric_idx);
                plt.title("{} (Best: {:.4f} on threshold {:.4f})".format(metnam, np.max(review[metric][2][cell]), 
                                                                         review["Threshold"][2][cell][np.argmax(review[metric][2][cell])]));
                plt.ylabel(metnam);
                plt.yscale("linear");
                plt.xlabel("Threshold");
                
                plt.plot(review["Threshold"][0][cell], review[metric][0][cell], label = "Train", color = [0.8, 0.2, 0.2, 1.0]);
                plt.scatter(review["Threshold"][0][cell], review[metric][0][cell], edgecolors = [0.8, 0.2, 0.2, 1.0], color = 'w', s = 10);
                plt.plot(review["Threshold"][1][cell], review[metric][1][cell], label = "Valid", color = [0.2, 0.8, 0.2, 1.0]);
                plt.scatter(review["Threshold"][1][cell], review[metric][1][cell], edgecolors = [0.2, 0.8, 0.2, 1.0], color = 'w', s = 10);
                plt.plot(review["Threshold"][2][cell], review[metric][2][cell], label = "Test", color = [0.2, 0.2, 0.8, 1.0]);
                plt.scatter(review["Threshold"][2][cell], review[metric][2][cell], edgecolors = [0.2, 0.2, 0.8, 1.0], color = 'w', s = 10);
                plt.legend();
            
            plt.subplot(4, 6, 19);
            plt.title("ROC Curve (AUROC. Valid: {:.4f} Test: {:.4f})".format(review["AUROC"][1][cell], review["AUROC"][2][cell]));
            plt.ylabel("TPR");
            plt.yscale("linear");
            plt.xlabel("FPR");
            plt.plot(review["FPR"][0][cell], review["Sensitivity"][0][cell], label = "Train", color = [0.8, 0.2, 0.2, 1.0]);
            plt.plot(review["FPR"][1][cell], review["Sensitivity"][1][cell], label = "Valid", color = [0.2, 0.8, 0.2, 1.0]);
            plt.plot(review["FPR"][2][cell], review["Sensitivity"][2][cell], label = "Test", color = [0.2, 0.2, 0.8, 1.0]);
            plt.plot([0, 1], [0, 1], label = "Random", color = [0.0, 0.0, 0.0, 1.0], linewidth = 0.7, linestyle = "dashed");
            plt.legend();

            plt.subplot(4, 6, 20);
            plt.title("PR Curve (AUPRC. Valid: {:.4f} Test: {:.4f})".format(review["AUPRC"][1][cell], review["AUPRC"][2][cell]));
            plt.ylabel("Precision");
            plt.yscale("linear");
            plt.xlabel("Recall");
            plt.plot(review["Sensitivity"][0][cell], review["Precision"][0][cell], label = "Train", color = [0.8, 0.2, 0.2, 1.0]);
            plt.plot(review["Sensitivity"][1][cell], review["Precision"][1][cell], label = "Valid", color = [0.2, 0.8, 0.2, 1.0]);
            plt.plot(review["Sensitivity"][2][cell], review["Precision"][2][cell], label = "Test", color = [0.2, 0.2, 0.8, 1.0]);
            plt.legend();
        
        ax = plt.subplot(4, 3, (11, 12));
        ax.set_title("Hit@K");
        ax.set_xlabel("K");
        ax.set_ylabel("Hit@K");
        x = np.arange(len(HITS));

        bar_width = 0.15;
        ax.bar(x - 2.5 * bar_width, review["Hit@K"][0][cell], width = bar_width * 0.9, label = "Train", color = [0.6, 0.2, 0.2, 1.0]);
        ax.bar(x - 0.5 * bar_width, review["Hit@K"][1][cell], width = bar_width * 0.9, label = "Valid", color = [0.2, 0.6, 0.2, 1.0]);
        ax.bar(x + 1.5 * bar_width, review["Hit@K"][2][cell], width = bar_width * 0.9, label = "Test", color = [0.2, 0.2, 0.6, 1.0]);

        ax.legend();
        ax.set_xticks(x);
        ax.set_xticklabels(HITS);

        for bar in ax.patches:
            bar_value = bar.get_height();
            text = "{:.3f}".format(bar_value);
            text_x = bar.get_x() +  bar.get_width() / 2;
            text_y = bar.get_y() + bar_value;
            bar_color = bar.get_facecolor();
            ax.text(text_x, text_y, text, ha = "center", va = "bottom", color = bar_color, size = 4);

    plt.savefig(dir_path + cell_names[cell] + pre_args.metric + ".jpg", dpi = 300);
    plt.clf();
    plt.close('all');