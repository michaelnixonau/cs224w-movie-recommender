import torch
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch_geometric.nn import LGConv
from torch_geometric.utils import to_undirected
from torch_geometric.data import HeteroData
from torch_geometric.nn import LGConv as PygLGConv

class LightGCN(nn.Module):
    def __init__(self, num_nodes, emb_dim=64, n_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.n_layers = n_layers

        self.emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        self.convs = nn.ModuleList([PygLGConv() for _ in range(n_layers)])

    def forward(self, edge_index):
        # edge_index: [2,E]
        x = self.emb.weight # (N, d)
        outs = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            outs.append(x)
        return torch.stack(outs, dim=0).mean(dim=0)
    

    def loss(self, user_emb, pos_emb, neg_emb, l2_reg=1e-4):
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        x = pos_scores - neg_scores
        bpr = F.softplus(-x).mean()

        reg = (user_emb.norm(dim=1).pow(2) +
            pos_emb.norm(dim=1).pow(2) +
            neg_emb.norm(dim=1).pow(2)).mean()
        return bpr + l2_reg * reg

    

    def fit(self, edge_index, num_users, sample_triples, train_set,
            optimizer, num_items, rng, epochs=100, batch_size=2048, l2_reg=1e-4, device="cuda", save_path = None):
        self.to(device)
        total_pos = sum(len(items) for items in train_set.values())
        steps = max(1, total_pos // batch_size)

        loss_history = []

        for epoch in tqdm(range(1, epochs + 1)):
            self.train()
            running = 0.0

            for _ in range(steps):
                all_emb = self.forward(edge_index)
                all_user_list, all_item_list = all_emb[:num_users], all_emb[num_users:]
                u, i, j = sample_triples(batch_size, train_set, num_users, num_items, device, rng)
                ue, ie, je = all_user_list[u], all_item_list[i], all_item_list[j]

                loss = self.loss(ue, ie, je, l2_reg=l2_reg)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running += float(loss.detach())
            loss_history.append(running / steps)

            if save_path is not None and epoch % 10 == 0:
                save_file = f"{save_path}_epoch{epoch}.pt"
                state_dict_cpu = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                torch.save(state_dict_cpu, save_file)
                # print(f"Saved checkpoint at epoch {epoch} -> {save_file}")

        # if save_path is not None:
        #     state_dict_cpu = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        #     torch.save(state_dict_cpu, save_path)

        return loss_history
    
    @torch.no_grad()
    def evaluate_full_ranking(
        self,
        edge_index: torch.Tensor,
        train_user_items: dict,
        test_pairs: list,
        num_users: int,
        num_items: int,
        device: str,
        K: int = 10,
        save_path: str = None
    ):
        self.eval()
        edge_index = edge_index.to(device)

        all_emb = self(edge_index)
        user_all, item_all = all_emb[:num_users], all_emb[num_users:]

        self.user_emb_cached = user_all.detach().cpu()
        self.item_emb_cached = item_all.detach().cpu()

        if save_path is not None:
            torch.save(
                {
                    "user_emb": self.user_emb_cached,
                    "item_emb": self.item_emb_cached,
                },
                save_path,
            )

        hr_hits, ndcg_sum, auc_sum, n_eval = 0, 0.0, 0.0, 0

        for (u, i_pos) in test_pairs:
            u_items = train_user_items.get(u, set())
            
            neg_items = [j for j in range(num_items) if (j not in u_items and j != i_pos)]

            cand_items = torch.tensor([i_pos] + neg_items, device=device, dtype=torch.long)

            u_emb = user_all[u].unsqueeze(0)
            i_emb = item_all[cand_items]
            scores = (u_emb * i_emb).sum(dim=1)

            # HR@K
            topk = min(K, scores.numel())
            topk_idx = torch.topk(scores, k=topk, dim=0).indices
            # 0 is the positive sample
            hit = (topk_idx == 0).any().item()
            if hit:
                hr_hits += 1
                ranks = torch.argsort(scores, descending=True)
                rank_pos = (ranks == 0).nonzero(as_tuple=False).item() + 1
                ndcg_sum += 1.0 / torch.log2(torch.tensor(rank_pos + 1.0, device=device)).item()

            # AUC：P(score_pos > score_neg)
            if scores.numel() > 1:
                auc = (scores[0] > scores[1:]).float().mean().item()
                auc_sum += auc

            n_eval += 1

        metrics = {
            "HR@K": hr_hits / max(1, n_eval),
            "NDCG@K": ndcg_sum / max(1, n_eval),
            "AUC": auc_sum / max(1, n_eval),
            "UsersEvaluated": n_eval,
        }
        return metrics
    
    @torch.no_grad()
    def compute_embeddings(
        self,
        edge_index: torch.Tensor,
        num_users: int,
        device: str = "cpu",
    ):
        """
        Compute LightGCN embeddings once and cache them for fast online recommendation.
        """
        self.eval()
        edge_index = edge_index.to(device)

        all_emb = self.forward(edge_index)
        user_all = all_emb[:num_users]
        item_all = all_emb[num_users:]

        self.user_emb_cached = user_all.detach().cpu()
        self.item_emb_cached = item_all.detach().cpu()

        return self.user_emb_cached, self.item_emb_cached
    
    @torch.no_grad()
    def recommend_topk(
        self,
        user_id: int,
        candidate_items,
        K: int = 10,
    ):
        """
        Recommend top-K items for a given user from a set of candidate_items
        candidate_items is the set of items that have not been interacted with user
        """
        if isinstance(candidate_items, torch.Tensor):
            cand = candidate_items.long()
        else:
            cand = torch.tensor(candidate_items, dtype=torch.long)

        user_emb = self.user_emb_cached[user_id]
        item_embs = self.item_emb_cached[cand]

        scores = (user_emb.unsqueeze(0) * item_embs).sum(dim=1)

        topk = min(K, scores.size(0))
        topk_scores, topk_idx = torch.topk(scores, k=topk, dim=0)

        topk_items = cand[topk_idx].tolist()
        return topk_items, topk_scores.cpu().numpy()
    
    @torch.no_grad()
    def recommend_one(
        self,
        user_id: int,
        candidate_items,
    ) -> int:
        topk_items, _ = self.recommend_topk(user_id, candidate_items, K=1)
        return topk_items[0]


