import numpy as np
from typing import Optional, Tuple
import pandas as pd
from collections import deque

from latent_static_env import GNNLatentStaticEnv


class LatentDynamicEnv(GNNLatentStaticEnv):
    """
    RecLab-style latent-dynamic environment. Environment with Feedback Loop.

    Based on latent-static, we add:
        - affinity: when user intearact with some item, latent move to embedding of that item
        - boredom: If more than t similar items are recommended in m recent recommendations, add penalty term
    The ground truth rating function between user u and item i is then: 
    r_t(u, i) = clip(mu + b_u + b_i + alpha <z_u^t, q_i> - boredom_penalty(u,i) + eps_t)
    eps_t ~ N(0, sigma^2) as uncertainty, z_u^t updates in timestamp t.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_emb,
        item_emb,
        ratings_df: pd.DataFrame,
        rng: np.random.RandomState,
        sigma_eps: float = 0.25,
        clip_min: float = 0.5,
        clip_max: float = 5.0,
        alpha: float = 1.0,
        affinity_rate: float = 0.05,
        boredom_lambda: float = 0.5,
        boredom_memory: int = 20,
        boredom_threshold: int = 3,
        similarity_threshold: float = 0.0001,
    ):
        if hasattr(user_emb, "detach"):
            user_emb = user_emb.detach().cpu().numpy()
        if hasattr(item_emb, "detach"):
            item_emb = item_emb.detach().cpu().numpy()

        latent_dim = 64
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            user_emb=user_emb,
            item_emb=item_emb,
            ratings_df=ratings_df,
            rng=rng,
            sigma=sigma_eps,
            clip_min=clip_min,
            clip_max=clip_max,
            alpha=alpha,
        )

        self.affinity_rate = float(affinity_rate)
        self.boredom_lambda = float(boredom_lambda)
        self.boredom_memory = int(boredom_memory)
        self.boredom_threshold = int(boredom_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.d = latent_dim

        self.user_history = [
            deque(maxlen=self.boredom_memory) for _ in range(self.num_users)
        ]

        norms = np.linalg.norm(self.user_emb, axis=1, keepdims=True) + 1e-8
        self._user_init_norms = norms.astype(np.float32)

    def _cosine_sim(self, x: np.ndarray, y: np.ndarray) -> float:
        denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-8
        return float(np.dot(x, y) / denom)

    def _boredom_penalty(self, u: int, i: int) -> float:
        if self.boredom_lambda <= 0 or self.boredom_memory <= 0:
            return 0.0

        hist = self.user_history[u]
        if len(hist) == 0:
            return 0.0

        q_i = self.item_emb[i]
        similar_cnt = 0

        for j in hist:
            q_j = self.item_emb[j]
            cos = self._cosine_sim(q_i, q_j)
            if cos >= self.similarity_threshold:
                similar_cnt += 1

        if similar_cnt >= self.boredom_threshold:
            return self.boredom_lambda
        return 0.0

    def _update_user_preference(self, u: int, i: int):
        """
        affinity update:
            z_u^{t+1} = (1 - η) z_u^t + η q_i
        """
        if self.affinity_rate <= 0.0:
            return

        z_u = self.user_emb[u]
        q_i = self.item_emb[i]

        new_z = (1.0 - self.affinity_rate) * z_u + self.affinity_rate * q_i
        new_norm = np.linalg.norm(new_z) + 1e-8
        target_norm = self._user_init_norms[u, 0]

        new_z = new_z * (target_norm / new_norm)
        self.user_emb[u] = new_z.astype(np.float32)

    def true_score_dynamic(self, u: int, i: int) -> float:
        base = super().true_score(u, i)
        penalty = self._boredom_penalty(u, i)
        return base - penalty

    def rate(self, u: int, i: int, update: bool = True) -> float:
        s = self.true_score_dynamic(u, i)
        eps = self.rng.normal(loc=0.0, scale=self.sigma)
        r = s + eps
        r = float(np.clip(r, self.clip_min, self.clip_max))

        if update:
            self._update_user_preference(u, i)
            self.user_history[u].append(i)

        return r

    def interact(self, u: int, i: int) -> float:
        return self.rate(u, i, update=True)

    def reset_user(self, u: int):
        self.user_emb[u] = self.rng.normal(
            loc=0.0,
            scale=np.linalg.norm(self.user_emb[u]) / np.sqrt(self.d),
            size=(self.d,)
        ).astype(np.float32)
        self.user_history[u].clear()

    def reset_all(self):
        self.user_history = [
            deque(maxlen=self.boredom_memory) for _ in range(self.num_users)
        ]