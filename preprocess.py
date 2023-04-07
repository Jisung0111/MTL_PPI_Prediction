import numpy as np
import os

if "PPI-Context.txt" not in os.listdir():
    os.system("wget https://github.com/montilab/ppi-context/raw/master/data/v_1_00/PPI-Context.txt");

CELLS = ["HeLa", "HEK293T", "HEK293", "HCT 116", "U2OS", "T-REx-293",
         "MCF-10A", "HeLa S3", "SH-SY5Y", "HEK", "Schneider 2", "BL-21", "MEF (C57BL/6)",
         "DU145", "MCF-7", "NIH 3T3", "MDA-MB-231", "K-562", "MRC-5", "293T/AT1",
         "FW3-47", "F56 [Mouse hybridoma]", "S10F03", "LNCaP", "BT-549"];
MERGE_CELLS = [["HEK293", "HCT 116"],
               ["HEK293T", "HEK293", "HCT 116"],
               ["HEK293T", "HEK293", "HCT 116", "U2OS"],
               ["HeLa", "HEK293T", "HCT 116", "U2OS"],
               ["HeLa", "HEK293", "HCT 116", "U2OS"],
               ["HEK293T", "HEK293", "T-REx-293"],
               ["HeLa", "HEK293T", "HEK293", "HCT 116", "U2OS"],
               ["HeLa", "HEK293T", "HEK293", "HCT 116", "U2OS", "T-REx-293"],
               ["HeLa", "HEK293T", "HEK293", "HCT 116", "U2OS", "T-REx-293",
                "MCF-10A", "HeLa S3", "SH-SY5Y", "HEK", "Schneider 2", "BL-21", "MEF (C57BL/6)",
                "DU145", "MCF-7", "NIH 3T3", "MDA-MB-231", "K-562", "MRC-5", "293T/AT1",
                "FW3-47", "F56 [Mouse hybridoma]", "S10F03", "LNCaP", "BT-549"]];
PMID_THRES1 = 26344197;
PMID_THRES2 = 28507161;
PMID_MAX = 30440001;
only_for_train = 6;

contents = [];
with open("PPI-Context.txt", 'r') as f:
    line = f.readline();
    while True:
        line = f.readline();
        if not line: break;
        contents.append(line.replace("\n","").split("\t"));
        
cell_data = {cell: {} for cell in CELLS};
for k in contents:
    genes = tuple(sorted([k[1], k[2]]));
    if "" in genes or k[4] not in CELLS: continue;
    cell_data[k[4]][genes] = min(cell_data[k[4]].get(genes, PMID_MAX), int(k[3]));

train_data = {cell: set() for cell in CELLS};
train_node = {cell: set() for cell in CELLS};

for k in contents:
    genes = tuple(sorted([k[1], k[2]]));
    if "" in genes or k[4] not in CELLS or PMID_THRES1 < cell_data[k[4]][genes]: continue;
    train_data[k[4]].add(genes);
    train_node[k[4]].add(k[1]);
    train_node[k[4]].add(k[2]);

valid_data = {cell: set() for cell in CELLS[:only_for_train]};
test_data = {cell: set() for cell in CELLS[:only_for_train]};

for k in contents:
    genes = tuple(sorted([k[1], k[2]]));
    if "" in genes or k[4] not in CELLS[:only_for_train] or cell_data[k[4]][genes] <= PMID_THRES1: continue;
    if k[1] not in train_node[k[4]] or k[2] not in train_node[k[4]]: continue;
    
    if cell_data[k[4]][genes] <= PMID_THRES2: valid_data[k[4]].add(genes);
    else: test_data[k[4]].add(genes);
    
for dir in ["Data", "Exps", "Results"]:
    if dir not in os.listdir("."): os.mkdir(dir);

for cell in CELLS:
    cell_name = cell.replace(" ","_").replace("/","_");
    with open("Data/{}.txt".format(cell_name), 'w') as f:
        for s, o in train_data[cell]: f.write("{}\t{}\n".format(s, o));
    
    if cell in CELLS[:only_for_train]:
        with open("Data/{}_valid.txt".format(cell_name), 'w') as f:
            for s, o in valid_data[cell]: f.write("{}\t{}\n".format(s, o));
        with open("Data/{}_test.txt".format(cell_name), 'w') as f:
            for s, o in test_data[cell]: f.write("{}\t{}\n".format(s, o));

for comb in MERGE_CELLS:
    accu_data = set();
    for cell in comb: accu_data |= train_data[cell];
    cell_name = " ".join([cell.replace(" ","_").replace("/","_") for cell in comb]);
    with open("Data/{}.txt".format(cell_name), 'w') as f:
        for s, o in accu_data: f.write("{}\t{}\n".format(s, o));
