import numpy as np
import torch as th
import torch.nn as nn
from collections import deque

class CollectMessage(th.autograd.Function):
    @staticmethod
    def forward(ctx, idcs, att, size):
        ctx.idcs = idcs[0];
        return th.stack([th.sparse.sum(th.sparse_coo_tensor(idcs, att[i], size), dim = 1).to_dense()
                         for i in range(att.shape[0])]);
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output[:, ctx.idcs, :] if ctx.needs_input_grad[1] else None, None;

def get_param(shape, zero = False):
    if zero: return nn.Parameter(th.zeros(shape));
    param = nn.Parameter(th.Tensor(*shape));
    nn.init.xavier_normal_(param.data);
    return param;

class Linear3d(nn.Module):
    def __init__(self, head, in_dim, out_dim):
        super(Linear3d, self).__init__();
        self.W = get_param([head, in_dim, out_dim]);
        self.b = nn.Parameter(th.zeros([head, 1, out_dim]));
        
    def forward(self, x):
        return x @ self.W + self.b;

class RAGATLayer(nn.Module):
    def __init__(self, num_rels, head, in_dim, out_dim, dropout):
        super(RAGATLayer, self).__init__();
        self.out_dim = out_dim;
        
        self.L_embed = get_param([1, in_dim]);
        self.Wrel = get_param([head, num_rels + 1, in_dim]);
        
        self.W1 = Linear3d(head, in_dim, out_dim);
        self.W0 = Linear3d(head, in_dim, out_dim);
        
        self.leaky_relu = nn.LeakyReLU(0.2);
        self.Wt = Linear3d(head, out_dim, 1);
        
        self.drop = nn.Dropout(dropout);
        self.bias = get_param([1, out_dim], True);
        self.bn = nn.BatchNorm1d(out_dim);
        self.act = th.tanh;
        
        self.Wr = nn.Linear(in_dim, out_dim);
    
    def forward(self, edge_idx, num_edges, N_embed, R_embed):
        num_nodes = N_embed.shape[0];
        # msg: head * (# edge + # node) * out_dim
        msg = th.cat([self.W1(self.Wrel[:, edge_idx[1, :num_edges], :] * N_embed[edge_idx[2, :num_edges]] * (R_embed[edge_idx[1, :num_edges]] + 1.0)),
                      self.W0(self.Wrel[:, edge_idx[1, num_edges:], :] * N_embed[edge_idx[2, num_edges:]] * (self.L_embed + 1.0))], dim = 1);
        # att: head * (# edge + # node) * 1
        att = th.exp(-self.leaky_relu(self.Wt(msg)));
        # att_sum: head * (# node) * 1;
        att_sum = CollectMessage.apply(edge_idx[::2], att, [num_nodes, num_nodes, 1]);
        att_sum[att_sum == 0.0] = 1.0;
        
        N_embed = self.drop(self.act(self.bn(
                        th.mean(CollectMessage.apply(edge_idx[::2],
                                                     self.drop(att) * msg,
                                                     [num_nodes, num_nodes, self.out_dim]
                                                    ) / att_sum, dim = 0) + \
                        self.bias
                    )));
        R_embed = self.Wr(R_embed);
        
        assert not th.isnan(N_embed).any();
        return N_embed, R_embed;
    
class DistMult(nn.Module):
    def __init__(self):
        super(DistMult, self).__init__();
    
    def forward(self, sub_emb, rel_emb):
        return sub_emb * rel_emb;

class ConvE(nn.Module):
    def __init__(self, num_channels, kernel_size, dim, dropout):
        super(ConvE, self).__init__();
        self.side_dim = int(pow(2 * dim, 0.5));
        self.conv = nn.Conv2d(1, num_channels, kernel_size);
        
        self.bn0 = nn.BatchNorm2d(1);
        self.bn1 = nn.BatchNorm2d(num_channels);
        self.bn2 = nn.BatchNorm1d(dim);
        self.relu = nn.ReLU(True);
        self.drop = nn.Dropout(dropout);
        
        self.flat_sz = num_channels * pow(self.side_dim - kernel_size + 1, 2);
        self.Wc = nn.Linear(self.flat_sz, dim);
    
    def forward(self, sub_emb, rel_emb):
        conv_input = th.stack([sub_emb, rel_emb], dim = 1).permute((0, 2, 1));
        conv_input = self.bn0(conv_input.reshape((-1, 1, self.side_dim, self.side_dim)));
        
        ret = self.relu(self.bn1(self.conv(conv_input)));
        ret = self.relu(self.bn2(self.drop(self.Wc(self.drop(ret).view((-1, self.flat_sz))))));
        return ret;

class LinkPredictBase(nn.Module):
    def __init__(self, edge_idx, node_idx, enc_param, dec_param):
        super(LinkPredictBase, self).__init__();
        self.num_nodes = enc_param["num_nodes"];
        self.num_rels = enc_param["num_rels"];
        self.num_edges = edge_idx.shape[1];
        self.node_idx = node_idx;
        
        self.edge_idx = np.concatenate([edge_idx,
                                        np.stack([np.arange(self.num_nodes),
                                                  np.full(self.num_nodes, self.num_rels, dtype = np.int64),
                                                  np.arange(self.num_nodes)])
                                       ], axis = 1);
        self.init_N_embed = get_param([self.num_nodes, enc_param["in_dim"]]);
        self.init_R_embed = get_param([self.num_rels, enc_param["in_dim"]]);
        
        self.layers = nn.ModuleList([RAGATLayer(self.num_rels, enc_param["head"], enc_param["in_dim"] if i == 0 else enc_param["out_dim"],
                                                enc_param["out_dim"], enc_param["dropout"])
                                     for i in range(enc_param["num_layers"])]);
        
        if self.num_rels == 1: self.decoder = lambda n, r: n;
        elif dec_param["decoder"] == "DistMult": self.decoder = DistMult();
        elif dec_param["decoder"] == "ConvE": self.decoder = ConvE(dec_param["num_channels"], dec_param["kernel_size"],
                                                                   enc_param["out_dim"], enc_param["dropout"]);
        else: raise ValueError("Decoder should be one of DistMult and ConvE.");
        
        self.bias = get_param([1, self.num_nodes], True);
    
    def get_embed(self):
        N_embed, R_embed = self.init_N_embed, self.init_R_embed;
        for layer in self.layers: N_embed, R_embed = layer(self.edge_idx, self.num_edges, N_embed, R_embed);
        return N_embed, R_embed;
    
    def forward(self, N_embed, R_embed, sub, r):
        srs = self.decoder(N_embed[sub], R_embed[r].repeat((sub.shape[0], 1)));
        score = th.sigmoid(srs @ N_embed.T + self.bias);
        return score;
    
    def eval_score(self, N_embed, R_embed, node_idx):
        srs = self.decoder(N_embed, R_embed.repeat((N_embed.shape[0], 1)));
        score = 0.5 * th.sigmoid(srs @ N_embed.T + self.bias[0, node_idx]);
        return score + score.T;

class LinkPredict:
    def __init__(self, edge_idx, node_idx, enc_param, dec_param, opt_param, lrn_param, device, pth_path = None):
        '''
        edge_idx: numpy([[s, ...], [r, ...], [o, ...]])
        enc_param: num_nodes, num_rels, num_layers, head, in_dim, out_dim, dropout
        dec_param: decoder(DistMult or ConvE), num_channels, kernel_size (ConvE)
        opt_param: lr, l2_reg, scheduler, scheduler parameter
        lrn_param: lbl_smooth, batch_size, grad_norm
        '''
        self.num_nodes, self.num_rels = enc_param["num_nodes"], enc_param["num_rels"];
        
        self.model = LinkPredictBase(edge_idx, node_idx, enc_param, dec_param);
        if pth_path is not None:
            model_load = th.load(pth_path, map_location = th.device("cpu"));
            self.model.load_state_dict(model_load);
        
        if device is not th.device("cpu"):
            self.model.cuda(device);
        
        if pth_path is None:
            self.optimizer = th.optim.AdamW(self.model.parameters(), lr = opt_param["lr"], weight_decay = opt_param["l2_reg"]);
            self.scheduler = opt_param["scheduler"];
            if self.scheduler == "CosineAnnealingWarmRestarts":
                self.lr_scheduler = th.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, T_0 = opt_param["T_0"], T_mult = opt_param["T_mult"], eta_min = opt_param["etamin"]);
            elif self.scheduler == "ReduceLROnPlateau":
                self.lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 'max', factor = opt_param["factor"], patience = opt_param["patience"], min_lr = opt_param["min_lr"]);
            elif self.scheduler == "None": self.lr_scheduler = None;
            else: raise ValueError("Invalid learning rate scheduler");
        
            self.lbl_smooth = lrn_param["lbl_smooth"];
            self.batch_size = lrn_param["batch_size"];
            self.grad_norm = lrn_param["grad_norm"];
            self.node_idx = node_idx;
            self.edge_list = [[] for _ in range(self.num_nodes * self.num_rels)];
            for s, r, o in edge_idx.T: self.edge_list[s + r * self.num_nodes].append(o);
            
            self.samples = [];
            for r, sub in enumerate(node_idx):
                for s in sub: self.samples.append(s + r * self.num_nodes);
            self.samples = np.array(self.samples);
            self.bceloss = th.nn.BCELoss();
    
    def label_vec(self, sub, r): # sub: numpy, r: int
        ret = [];
        for s in sub:
            lbl = th.zeros(self.num_nodes);
            for o in self.edge_list[s + r * self.num_nodes]: lbl[o] = 1.0;
            if self.lbl_smooth > 0.0: lbl = (1.0 - self.lbl_smooth) * lbl + (1.0 / self.num_nodes);
            ret.append(lbl);
        return th.stack(ret);
    
    def run_epoch(self, epoch):
        num_iter = self.samples.shape[0] // self.batch_size;
        losses = [];
        
        while True:
            valid = True;
            idcs = self.samples[np.random.permutation(self.samples.shape[0])];
            for iter in range(num_iter):
                idx = idcs[iter * self.batch_size: (iter + 1) * self.batch_size];
                if 0 < np.sum(np.unique(idx // self.num_nodes, return_counts = True)[1] < 3):
                    valid = False; break;
            if valid: break;
        
        self.model.train();
        for iter in range(num_iter):
            self.optimizer.zero_grad();
            
            idx = idcs[iter * self.batch_size: (iter + 1) * self.batch_size];
            rel = idx // self.num_nodes;
            r_sub = {r: [] for r in np.unique(rel)};
            for s, r in zip(idx % self.num_nodes, rel): r_sub[r].append(s);
            
            loss = 0;
            N_embed, R_embed = self.model.get_embed();
            for r, sub in r_sub.items():
                sub = np.array(sub);
                scores = self.model(N_embed, R_embed, sub, r);
                labels = self.label_vec(sub, r).to(scores.device);
                loss = loss + self.bceloss(scores, labels);
            loss = loss / len(r_sub);
            
            with th.cuda.device(scores.device): th.cuda.empty_cache();
            
            loss.backward();
            th.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm);
            self.optimizer.step();
            
            if self.scheduler == "CosineAnnealingWarmRestarts":
                self.lr_scheduler.step(epoch + (iter + 1) // num_iter);
            
            losses.append(loss.detach().item());
        
        loss = np.mean(losses);
        return epoch + 1, loss;
        
    def evaluate(self, edge_list, edge_ulist, node_idx, cell_idx, AUROC_ive = 10000, hits = [1, 10, 100],
                 review = False, save_score = False, dist_thres = 101):
        '''
        edge_list: list(3(split), #rel, numpy([[s', ...], [o, ...]]))
        edge_ulist: list(3(split), #rel, numpy([[s', ...], [o, ...]]))
                    contains edges on the upper triangle
        node_idx: list(#rel, numpy(#node_r));
        cell_idx: list(#rel)
        AUROC_ive: inverse of maximum error allowed for AUROC calculation 
        hits: K's of Hit@K
        review: True if reviewer
        save_score: True if you want to save scores. (need CPU memory)
        dist_thres: uniformly divided threshold for Score Distribution
        
        output
        if not review(default)
            Hit@K, MRR, MR, AUROC
        if review
            Hit@K, MRR, MR, AUROC, AUPRC,
            Score_Distribution(dist_thres),
            Accuracy, Sensitivity, Specificity, FPR, Precision, MCC, F1_Score, Threshold,
            Score(if save_score)
        
        Hit@K: list(3(split), rel + ALL, K)
        MRR: list(3(split), rel + ALL)
        MR: list(3(split), rel + ALL)
        Score_Dist: list(4(split + REST), rel + ALL, dist_thres) # number of edges s.t. score <= thres
        Accuracy, Sensitivity, Specificity, FPR, Precision, MCC, F1_Score, Threshold: list(3(split), rel, #thres)
        Score: list(rel, numpy(#nodes, #nodes))
        '''
        
        num_rels = len(cell_idx);
        eval_result = {
            "Hit@K"    : [[[0 for hit in hits] for rel in range(num_rels + 1)] for split in range(3)],
            "MRR"      : [[0 for rel in range(num_rels + 1)] for split in range(3)],
            "MR"       : [[0 for rel in range(num_rels + 1)] for split in range(3)],
            "AUROC"      : [[0 for rel in range(num_rels + 1)] for split in range(3)]
        };
        
        num_rest_edges = [node_idx[rel].shape[0] * (node_idx[rel].shape[0] + 1) // 2 - \
                          sum([edge_ulist[split][rel].shape[1] for split in range(3)]) for rel in range(num_rels)];
        num_edges = [[edge_ulist[split][rel].shape[1] for split in range(3)] for rel in range(num_rels)];
        FN = [[{} for split in range(3)] for rel in range(num_rels)];
        FP = [{} for rel in range(num_rels)];
        
        if review:
            Dthres = th.linspace(0.0, 1.0, dist_thres);
            eval_result.update({
                "Score_Dist": [[[0 for thres in Dthres] for rel in range(num_rels + 1)] for split in range(4)],
                "AUPRC"      : [[0 for rel in range(num_rels + 1)] for split in range(3)],
                "Accuracy"   : [[[] for rel in range(num_rels)] for split in range(3)],
                "Sensitivity": [[[] for rel in range(num_rels)] for split in range(3)],
                "Specificity": [[[] for rel in range(num_rels)] for split in range(3)],
                "FPR"        : [[[] for rel in range(num_rels)] for split in range(3)],
                "Precision"  : [[[] for rel in range(num_rels)] for split in range(3)],
                "MCC"        : [[[] for rel in range(num_rels)] for split in range(3)],
                "F1_Score"   : [[[] for rel in range(num_rels)] for split in range(3)],
                "Threshold"  : [[None for rel in range(num_rels)] for split in range(3)]
            });
            
            if save_score: eval_result.update({"Score": [None for rel in range(num_rels)]});
        
        self.model.eval();
        with th.no_grad():
            N_embed, R_embed = self.model.get_embed();
            
            for rel in range(num_rels):
                scores = self.model.eval_score(N_embed[node_idx[rel]], R_embed[cell_idx[rel]], node_idx[rel]);
                s_idx = [edge_list[split][rel][0] for split in range(3)];
                o_idx = [edge_list[split][rel][1] for split in range(3)];
                split_score = [scores[s_idx[split], o_idx[split]] for split in range(3)];
                
                usplit_score = [th.sort(scores[edge_ulist[split][rel][0], edge_ulist[split][rel][1]])[0].tolist() for split in range(3)];
                usort_score = [sorted(list(set(usplit_score[split] + [0.0, 1.0]))) for split in range(3)];
                
                if save_score: eval_result["Score"][rel] = scores.cpu().numpy();
                
                for split in range(3): scores[s_idx[split], o_idx[split]] = 0.0;
                
                max_rank = scores.shape[1] + 1;
                for split in range(3):
                    scores[s_idx[split], o_idx[split]] = split_score[split];
                    
                    with th.cuda.device(N_embed.device): th.cuda.empty_cache();
                    score_rank = 1 + th.argsort(th.argsort(scores, descending = True));
                    
                    sr_rank = [max_rank for _ in range(scores.shape[0])];
                    for s, o in zip(s_idx[split], o_idx[split]): sr_rank[s] = min(sr_rank[s], score_rank[s, o].item());
                    sr_rank = np.array([s for s in sr_rank if s != max_rank], dtype = np.float64);
                    
                    for hit_i, hit in enumerate(hits): eval_result["Hit@K"][split][rel][hit_i] = np.mean(sr_rank <= hit);
                    eval_result["MRR"][split][rel] = np.mean(1.0 / sr_rank);
                    eval_result["MR"][split][rel] = np.mean(sr_rank);
                    
                    scores[s_idx[split], o_idx[split]] = 0.0;
                
                triu_scores = scores.triu();
                triu_scores = triu_scores[triu_scores.nonzero(as_tuple = True)];
                triu_scores = th.cat((th.zeros(num_rest_edges[rel] - triu_scores.shape[0], device = triu_scores.device),
                                      triu_scores));
                
                if review:
                    for i, thres in enumerate(Dthres):
                        eval_result["Score_Dist"][3][rel][i] = th.sum(triu_scores <= thres).item();
                        eval_result["Score_Dist"][3][-1][i] += eval_result["Score_Dist"][3][rel][i];
                        for split in range(3):
                            eval_result["Score_Dist"][split][rel][i] = np.sum(np.array(usplit_score[split]) <= thres.item());
                            eval_result["Score_Dist"][split][-1][i] += eval_result["Score_Dist"][split][rel][i];
                
                triu_scores = th.sort(triu_scores)[0].tolist();
                
                for split in range(3):
                    idx_thres = [0.0, 1.0];
                    FN, FP = [0, num_edges[rel][split]], [num_rest_edges[rel], 0];
                    dq = deque([(0, len(usort_score[split]) - 1, 0, num_edges[rel][split], num_rest_edges[rel], 0, AUROC_ive)]);
                    M = num_edges[rel][split] * num_rest_edges[rel] * 2;
                    
                    while dq:
                        l, r, lfn, rfn, lfp, rfp, ive = dq.popleft();
                        
                        m = (l + r) // 2; t = usort_score[split][m];
                        idx_thres.append(t);
                        bl, br = 0, num_edges[rel][split] - 1;
                        while bl <= br:
                            bm = (bl + br) // 2;
                            if usplit_score[split][bm] <= t: bl = bm + 1;
                            else: br = bm - 1;
                        FN.append(bl);
                        
                        bl, br = 0, num_rest_edges[rel] - 1;
                        while bl <= br:
                            bm = (bl + br) // 2;
                            if triu_scores[bm] > t: br = bm - 1;
                            else: bl = bm + 1;
                        FP.append(num_rest_edges[rel] - bl);
                        
                        fn, fp = FN[-1], FP[-1];
                        
                        if l + 1 == m:
                            lt = usort_score[split][l];
                            bl, br = 0, num_rest_edges[rel] - 1;
                            while bl <= br:
                                bm = (bl + br) // 2;
                                if triu_scores[bm] > lt: br = bm - 1;
                                else: bl = bm + 1;
                            
                            if bl != num_rest_edges[rel] and triu_scores[bl] < t:
                                lt = triu_scores[bl];
                                idx_thres.append(lt);
                                FN.append(fn);
                                
                                br = num_rest_edges[rel] - 1;
                                while bl <= br:
                                    bm = (bl + br) // 2;
                                    if triu_scores[bm] > lt: br = bm - 1;
                                    else: bl = bm + 1;
                                FP.append(num_rest_edges[rel] - bl);
                            el = 0;
                        else: el = abs((lfn - fn) * (lfp - fp)) * ive;
                        
                        if m + 1 == r:
                            rt = usort_score[split][r];
                            bl, br = 0, num_rest_edges[rel] - 1;
                            while bl <= br:
                                bm = (bl + br) // 2;
                                if triu_scores[bm] > t: br = bm - 1;
                                else: bl = bm + 1;
                            
                            if bl != num_rest_edges[rel] and triu_scores[bl] < rt:
                                rt = triu_scores[bl];
                                idx_thres.append(rt);
                                FN.append(rfn);
                                
                                br = num_rest_edges[rel] - 1;
                                while bl <= br:
                                    bm = (bl + br) // 2;
                                    if triu_scores[bm] > rt: br = bm - 1;
                                    else: bl = bm + 1;
                                FP.append(num_rest_edges[rel] - bl);
                            er = 0;
                        else: er = abs((rfn - fn) * (rfp - fp)) * ive;
                        
                        if M < el + er:
                            if M >= 2 * el:
                                if m != r: dq.append((m, r, fn, rfn, fp, rfp, ive * M // (M - el) + 1));
                            elif M >= 2 * er:
                                if l != m: dq.append((l, m, lfn, fn, lfp, fp, ive * M // (M - er) + 1));
                            else:
                                ive *= el + er;
                                if m != r: dq.append((m, r, fn, rfn, fp, rfp, ive // er + 1));
                                if l != m: dq.append((l, m, lfn, fn, lfp, fp, ive // el + 1));
                    
                    
                    arri = sorted([i for i in range(len(idx_thres))], key = lambda i: idx_thres[i]);
                    arr = [(FP[i], FN[i]) for i in arri];
                    
                    eval_result["AUROC"][split][rel] = 1.0 - sum([
                       (arr[i][0] - arr[i + 1][0]) * (arr[i][1] + arr[i + 1][1]) for i in range(len(arr) - 1)
                    ]) / M;
                    
                    if review:
                        arr = [(fn, (num_edges[rel][split] - fn) / (num_edges[rel][split] - fn + fp)) for fp, fn in arr 
                               if num_edges[rel][split] - fn + fp];
                        eval_result["AUPRC"][split][rel] = sum([
                            (arr[i + 1][0] - arr[i][0]) * (arr[i + 1][1] + arr[i][1]) for i in range(len(arr) - 1)
                        ]) / (2 * num_edges[rel][split]);
                        
                        eval_result["Threshold"][split][rel] = [idx_thres[i] for i in arri];
                        for i in arri:
                            fn, fp = FN[i], FP[i];
                            tp, tn = num_edges[rel][split] - fn, num_rest_edges[rel] - fp;
                            eval_result["Accuracy"][split][rel].append((tp + tn) / (tp + fn + fp + tn));
                            eval_result["Sensitivity"][split][rel].append(tp / (tp + fn));
                            eval_result["Specificity"][split][rel].append(tn / (tn + fp));
                            eval_result["FPR"][split][rel].append(fp / (tn + fp));
                            eval_result["Precision"][split][rel].append(tp / (tp + fp) if tp + fp else float("nan"));
                            eval_result["MCC"][split][rel].append((tp * tn - fp * fn) / pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5) if \
                                                                  (tp + fp) * (tn + fn) else 0.0);
                            eval_result["F1_Score"][split][rel].append(2 * tp / (2 * tp + fp + fn));
                
                del triu_scores, scores;
            
        for split in range(3):
            for hit_i, hit in enumerate(hits):
                eval_result["Hit@K"][split][-1][hit_i] = np.mean([eval_result["Hit@K"][split][rel][hit_i] for rel in range(num_rels)]);
            eval_result["MRR"][split][-1] = np.mean([eval_result["MRR"][split][rel] for rel in range(num_rels)]);
            eval_result["MR"][split][-1] = np.mean([eval_result["MR"][split][rel] for rel in range(num_rels)]);
            eval_result["AUROC"][split][-1] = np.mean([eval_result["AUROC"][split][rel] for rel in range(num_rels)]);
            if review: eval_result["AUPRC"][split][-1] = np.mean([eval_result["AUPRC"][split][rel] for rel in range(num_rels)]);
        
        return eval_result;
    
    def save(self, path):
        th.save(self.model.state_dict(), path);
    
    