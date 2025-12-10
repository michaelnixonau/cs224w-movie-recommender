import numpy as np
from typing import Optional, Tuple
import pandas as pd
import torch


class GNNLatentStaticEnv:
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_emb = None,
        item_emb = None,
        ratings_df: pd.DataFrame = None,
        rng = None,
        sigma: float = 0.25,
        clip_min: float = 0.5,
        clip_max: float = 5.0,
        alpha: float = 1.0,
        latent_dim: int = 64,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.rng = rng if rng is not None else np.random.RandomState(42)
        self.d = latent_dim

        def to_numpy_or_none(x):
            if x is None:
                return None
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy().astype(np.float32)
            return np.array(x, dtype=np.float32)

        user_emb = to_numpy_or_none(user_emb)
        item_emb = to_numpy_or_none(item_emb)

        if user_emb is None:
            self.user_emb = self.rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(latent_dim),
                size=(num_users, latent_dim),
            ).astype(np.float32)
        else:
            self.user_emb = user_emb

        if item_emb is None:
            self.item_emb = self.rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(latent_dim),
                size=(num_items, latent_dim),
            ).astype(np.float32)
        else:
            self.item_emb = item_emb

        self.sigma = sigma
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.alpha = alpha
        self.ratings = ratings_df

        self._init_biases_from_ratings(ratings_df)



    def _init_biases_from_ratings(self, ratings_df: "pd.DataFrame"):
        """
        Estimate mu (population mean), user_bias, item_bias from observed ratings.
        Assume ratings_df.user_id and item_id are 0-based index with same index from embeddings.
        """
        self.mu = ratings_df["rating"].astype(int).mean()

        # user_bias: mean_u - mu
        user_mean = ratings_df.groupby("user_idx")["rating"].mean()
        self.user_bias = np.zeros(self.num_users, dtype=np.float32)
        self.user_bias[user_mean.index.values] = (user_mean.values - self.mu).astype(np.float32)

        # item_bias: mean_i - mu
        item_mean = ratings_df.groupby("item_idx")["rating"].mean()
        self.item_bias = np.zeros(self.num_items, dtype=np.float32)
        self.item_bias[item_mean.index.values] = (item_mean.values - self.mu).astype(np.float32)


    def true_score(self, u: int, i: int) -> float:
        """
        Caculate the predicted rating (without noise and clip):
        s(u,i) = mu + b_u + b_i + alpha * <e_u, e_i>
        """

        dot = self.alpha * float(np.dot(self.user_emb[u], self.item_emb[i]))
        s = self.mu + self.user_bias[u] + self.item_bias[i] + dot
        return s

    def rate(self, u: int, i: int) -> float:
        """
        Sample a noisy rating:
            r(u,i) = clip( true_score(u,i) + epsilon )
        where epsilon ~ N(0, sigma^2)
        """
        s = self.true_score(u, i)
        eps = self.rng.normal(loc=0.0, scale=self.sigma)
        r = s + eps
        r = float(np.clip(r, self.clip_min, self.clip_max))
        return r


    def sample_random_user(self) -> int:
        return int(self.rng.randint(0, self.num_users))

    def sample_random_item(self) -> int:
        return int(self.rng.randint(0, self.num_items))
    
    def sample_random_item_per_user(self, user_index: int) -> int:
        """
        Sample pairwise preference for user u between items randomly chosen item pair.
        Returns y (boolean) and optional p (probability)
        """
        i = self.sample_random_item()
        
        return self.rate(user_index, i)

    def alpha_fine_tune(
        self,
        alpha_min: float = 0.0,
        alpha_max: float = 0.2,
        num_alpha: int = 21,
        max_samples: int = 100000,
        num_check: int = 50000,
    ) -> Tuple[float, float]:
        """
        Finetune 'alpha' and 'sigma' (std of Gaussian noise ε) by matching the simulated distribution
        real distribution.
        Find the alpha that minimize the residual while keeping total variance same.
        """
        results = []
        df = self.ratings

        df_sample = df.sample(n=max_samples, random_state=42)

        u_idx = df_sample["user_idx"].to_numpy(dtype=np.int64)
        i_idx = df_sample["item_idx"].to_numpy(dtype=np.int64)
        ratings = df_sample["rating"].to_numpy(dtype=np.float32)

        r_min = int(ratings.min())
        r_max = int(ratings.max())
        bins = np.arange(r_min - 0.5, r_max + 1.5, 1.0)
        num_levels = r_max - r_min + 1

        hist_real, _ = np.histogram(ratings, bins=bins, density=True)
        print("Real rating density:")
        for k in range(num_levels):
            level = r_min + k
            print(f"  {level}: {hist_real[k]:.4f}")

        bias_term = (
            self.mu
            + self.user_bias[u_idx]
            + self.item_bias[i_idx]
        ).astype(np.float32)

        residual = ratings - bias_term

        z = np.sum(self.user_emb[u_idx] * self.item_emb[i_idx], axis=1).astype(np.float32) # dot product

        sigma_R = float(residual.std())
        sigma_z = float(z.std())
        print(f"\nVar(R)={sigma_R**2:.4f}, Var(z)={sigma_z**2:.4f}")

        def hist_distance(p: np.ndarray, q: np.ndarray) -> float:
            return float(np.sqrt(((p - q) ** 2).sum()))

        alphas = np.linspace(alpha_min, alpha_max, num_alpha)
        best_alpha = None
        best_sigma_eps = None
        best_dist = float("inf")

        if len(df_sample) > num_check:
            idx_all = self.rng.choice(len(df_sample), size=num_check, replace=False)
        else:
            idx_all = np.arange(len(df_sample))

        z_all = z[idx_all]
        bias_all = bias_term[idx_all]

        for alpha in alphas:
            # To keep two distributions have same varaince, need to have:
            # sigma_eps = np.sqrt(sigma_R**2 - alpha**2 * sigma_z**2)
            if alpha**2 * sigma_z**2 >= sigma_R**2:
                continue
            sigma_eps = np.sqrt(sigma_R**2 - alpha**2 * sigma_z**2)

            eps_sim = self.rng.normal(loc=0.0, scale=sigma_eps, size=z_all.shape[0])
            ratings_sim = bias_all + alpha * z_all + eps_sim
            ratings_sim = np.clip(ratings_sim, r_min, r_max)

            hist_sim, _ = np.histogram(ratings_sim, bins=bins, density=True)
            dist = hist_distance(hist_real, hist_sim)

            # print(f"alpha={alpha:.3f}, sigma_eps={sigma_eps:.3f}, hist_dist={dist:.5f}")
            results.append({
                "alpha": alpha,
                "sigma_eps": sigma_eps,
                "hist_real": hist_real,
                "hist_sim": hist_sim,
                "hist_dist": dist,
            })

            if dist < best_dist:
                best_dist = dist
                best_alpha = alpha
                best_sigma_eps = sigma_eps

        print("\nChosen parameters by histogram matching")
        print(f"best_alpha = {best_alpha:.4f}")
        print(f"best_sigma_eps = {best_sigma_eps:.4f}")
        print(f"best_hist_dist = {best_dist:.5f}")

        eps_best = self.rng.normal(loc=0.0, scale=best_sigma_eps, size=z_all.shape[0])
        ratings_best = bias_all + best_alpha * z_all + eps_best
        ratings_best = np.clip(ratings_best, r_min, r_max)

        print("\nSimulated ratings with best alpha/sigma (hist matching)")
        print("mean =", float(ratings_best.mean()))
        print("std =", float(ratings_best.std()))

        hist_best, _ = np.histogram(ratings_best, bins=bins, density=True)
        print("\nBest histogram vs real")
        for k in range(num_levels):
            level = r_min + k
            print(f"{level}: real={hist_real[k]:.4f}, sim={hist_best[k]:.4f}")

        self.alpha = float(best_alpha)
        self.sigma = float(best_sigma_eps)
        eval_results = pd.DataFrame(results)
        return self.alpha, self.sigma, eval_results