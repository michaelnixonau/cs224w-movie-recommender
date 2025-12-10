import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected
from torch_geometric.data import HeteroData

from data_preprocessing import sample_triples


class GraphAttentionNet(nn.Module):
    def __init__(self, num_nodes, emb_dim=64, n_layers=3, heads=1):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.heads = heads

        self.emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)

        self.convs = nn.ModuleList(
            [GATConv(emb_dim, emb_dim, heads=heads, concat=False) for _ in range(n_layers)]
        )

    def forward(self, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight
        outs = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)
            outs.append(x)

        return torch.stack(outs, dim=0).mean(dim=0)

    def loss(self, user_emb, pos_emb, neg_emb, l2_reg=1e-4):
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        x = pos_scores - neg_scores
        bpr = F.softplus(-x).mean()

        reg = (
            user_emb.norm(dim=1).pow(2)
            + pos_emb.norm(dim=1).pow(2)
            + neg_emb.norm(dim=1).pow(2)
        ).mean()
        return bpr + l2_reg * reg

    def fit(
        self,
        edge_index,
        num_users,
        sample_triples,
        train_set,
        optimizer,
        num_items,
        rng,
        epochs=100,
        batch_size=2048,
        l2_reg=1e-4,
        device="cuda",
        save_path=None,
    ):
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

                u, i, j = sample_triples(
                    batch_size, train_set, num_users, num_items, device, rng
                )
                ue, ie, je = all_user_list[u], all_item_list[i], all_item_list[j]

                loss = self.loss(ue, ie, je, l2_reg=l2_reg)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running += float(loss.detach())

            loss_history.append(running / steps)

            if save_path is not None and epoch % 10 == 0:
                save_file = f"{save_path}_epoch{epoch}.pt"
                state_dict_cpu = {
                    k: v.detach().cpu() for k, v in self.state_dict().items()
                }
                torch.save(state_dict_cpu, save_file)

        return loss_history

    @torch.no_grad()
    def compute_embeddings(
        self,
        edge_index: torch.Tensor,
        num_users: int,
        device: str = "cpu",
        save_path=None,
    ):
        self.eval()
        edge_index = edge_index.to(device)

        all_emb = self.forward(edge_index)
        user_all = all_emb[:num_users]
        item_all = all_emb[num_users:]

        self.user_emb_cached = user_all.detach().cpu()
        self.item_emb_cached = item_all.detach().cpu()

        if save_path is not None:
            save_dict = {
                "user_emb": self.user_emb_cached,
                "item_emb": self.item_emb_cached,
            }
            torch.save(save_dict, save_path)
            print(f"[Saved embeddings] {save_path}")

        return self.user_emb_cached, self.item_emb_cached

    @torch.no_grad()
    def recommend_topk(
        self,
        user_id: int,
        candidate_items,
        K: int = 10,
    ):
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


def retrain_model_from_df(
    combined_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    device: torch.device,
    rating_threshold: float = 3.5,
    emb_dim: int = 64,
    n_layers: int = 3,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 2048,
    l2_reg: float = 1e-4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:

    rng = np.random.RandomState(seed)

    df_pos = combined_df[combined_df["rating"] >= rating_threshold].copy()

    u_np = df_pos["user_idx"].to_numpy(dtype=np.int64)
    i_np = df_pos["item_idx"].to_numpy(dtype=np.int64)

    train_set = {u: set() for u in range(num_users)}
    for u, i in zip(u_np, i_np):
        train_set[int(u)].add(int(i))

    train_u = u_np
    train_i = i_np

    src = torch.tensor(train_u, dtype=torch.long)
    dst = torch.tensor(train_i + num_users, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)

    model = GraphAttentionNet(
        num_nodes=num_users + num_items,
        emb_dim=emb_dim,
        n_layers=n_layers,
        heads=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    _ = model.fit(
        edge_index=edge_index,
        num_users=num_users,
        sample_triples=sample_triples,
        train_set=train_set,
        optimizer=optimizer,
        num_items=num_items,
        rng=rng,
        epochs=epochs,
        batch_size=batch_size,
        l2_reg=l2_reg,
        device=device,
        save_path=None,
    )

    user_emb, item_emb = model.compute_embeddings(
        edge_index=edge_index,
        num_users=num_users,
        device=device,
    )

    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    return user_emb, item_emb