import numpy as np
import torch
import pandas as pd
from collections import defaultdict

from latent_static_env import GNNLatentStaticEnv
from latent_dynamic_env import LatentDynamicEnv
from lightgcn import retrain_model_from_df


def sample_offline_from_env(
    env: GNNLatentStaticEnv,
    ratings_df: pd.DataFrame,
    num_samples: int,
    rng: np.random.RandomState,
):
    """
    Sample (user, item) pairs from the ratings DataFrame and generate corresponding rating by latent env.
    """
    n = len(ratings_df)
    if n >= num_samples:
        df_sample = ratings_df.sample(
            n=num_samples,
            replace=False,
            random_state=rng.randint(0, 10**9),
        )
    else:
        df_sample = ratings_df.sample(
            n=num_samples,
            replace=True,
            random_state=rng.randint(0, 10**9),
        )

    u_idx = df_sample["user_idx"].to_numpy(dtype=np.int64)
    i_idx = df_sample["item_idx"].to_numpy(dtype=np.int64)

    ratings = []
    for u, i in zip(u_idx, i_idx):
        r = env.rate(int(u), int(i))
        ratings.append(r)

    ratings = np.array(ratings, dtype=np.float32)
    return u_idx, i_idx, ratings



def build_train_user_items_from_ratings(user_ids, item_ids, ratings, thresh=3.5):
    """
    Build the rating dictionary where items with rating >= threshold are considered as positive interactions.
    """
    train_user_items = defaultdict(set)
    for u, i, r in zip(user_ids, item_ids, ratings):
        if r >= thresh:
            train_user_items[int(u)].add(int(i))
    return train_user_items


def recommend_one_item_for_user_from_emb(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    train_user_items,
    user_idx: int,
    device: torch.device,
):
    """
    Recommend one highest-scoring unseen item for a given user based on embeddings from lightgcn.
    """
    num_items = item_emb.size(0)
    seen_items = train_user_items.get(user_idx, set())
    if len(seen_items) >= num_items:
        return None

    u_vec = user_emb[user_idx].to(device)
    scores = torch.matmul(item_emb.to(device), u_vec)

    if seen_items:
        seen_idx = torch.tensor(list(seen_items), dtype=torch.long, device=device)
        scores[seen_idx] = -1e9

    rec_item = int(torch.argmax(scores).item())
    return rec_item

def oracle_mean_rating_for_user(env, user_idx: int, num_items: int) -> float:
        """
        Approximate oracle mean rating for a user u over all items.
        We use deterministic rating here.
        """
        best = -np.inf

        for i in range(num_items):
            r = env.true_score(user_idx, i)
            if r > best:
                best = r
        return float(best)


def run_online_simulation_static(
    emb_path=r"embeddings/lightgcn_embeddings_ml_latest_small.pt",
    save_path = None,
    num_users=None,
    num_items=None,
    num_init_ratings=100_000,
    num_online_steps=1_000,
    users_per_step=200,
    rating_threshold=3.5,
    device_str="cuda"
):
    """
    Run online simulation based on following steps:
        1: Load lightgcn embeddings.
        2. At each step recommend items to a batch of users.
        3. Query env to calculate ratings for recommendation.
        4. Calculate batch means/regrets/coverage/diversity.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    rng = np.random.RandomState(42)

    embeddings = torch.load(emb_path, map_location=device)
    user_emb, item_emb = embeddings['user_emb'], embeddings['item_emb']
    num_users, num_items = user_emb.shape[0], item_emb.shape[0]

    if num_users is None:
        num_users = user_emb.size(0)
    if num_items is None:
        num_items = item_emb.size(0)

    assert user_emb.size(0) == num_users
    assert item_emb.size(0) == num_items


    rating_df = pd.read_csv(r'data/ml-latest-small/ratings.csv')
    user_ids, users = pd.factorize(rating_df['userId'])
    item_ids, items = pd.factorize(rating_df['movieId'])
    rating_df["user_idx"] = user_ids
    rating_df["item_idx"] = item_ids

    env = GNNLatentStaticEnv(
        num_users=num_users,
        num_items=num_items,
        user_emb=None,
        item_emb=None,
        ratings_df=rating_df,
        rng=rng,
        sigma=0.25,
        clip_min=0.5,
        clip_max=5.0,
        alpha=0.1,
    )

    init_users, init_items, init_ratings = sample_offline_from_env(
        env=env,
        ratings_df=rating_df,
        num_samples=num_init_ratings,
        rng=rng,
    )

    # train_user_items = build_train_user_items_from_ratings(
    #     init_users, init_items, init_ratings, thresh=rating_threshold
    # )
    train_user_items = defaultdict(set)


    online_mean_ratings = []
    online_mean_regrets = []
    step_distinct_items = []
    step_cum_coverage = []

    global_recommended_items = set()
    item_recommend_counts = np.zeros(num_items, dtype=np.int64)

    for step in range(1, num_online_steps + 1):
        batch_users = rng.choice(num_users, size=min(users_per_step, num_users), replace=False)

        step_ratings = []
        step_regrets = []
        step_items = []

        for u in batch_users:
            u = int(u)

            rec_item = recommend_one_item_for_user_from_emb(
                user_emb=user_emb,
                item_emb=item_emb,
                train_user_items=train_user_items,
                user_idx=u,
                device=device,
            )
            if rec_item is None:
                continue

            rec_item = int(rec_item)

            r = float(env.rate(u, rec_item))
            step_ratings.append(r)

            if r >= rating_threshold:
                if rec_item not in train_user_items[u]:
                    train_user_items[u].add(rec_item)

            oracle_r = oracle_mean_rating_for_user(env, u, num_items)
            step_regrets.append(oracle_r - r)

            step_items.append(rec_item)
            global_recommended_items.add(rec_item)
            item_recommend_counts[rec_item] += 1

        mean_r = float(np.mean(step_ratings)) if len(step_ratings) > 0 else np.nan
        mean_regret = float(np.mean(step_regrets)) if len(step_regrets) > 0 else np.nan
        online_mean_ratings.append(mean_r)
        online_mean_regrets.append(mean_regret)

        distinct_cnt = len(set(step_items))
        step_distinct_items.append(distinct_cnt)

        cum_cov = len(global_recommended_items) / float(num_items)
        step_cum_coverage.append(cum_cov)



        print(
            f"[step {step:4d}] "
            f"mean rating = {mean_r:.4f}, "
            f"mean regret = {mean_regret:.4f}, "
            f"distinct items = {distinct_cnt:3d}, "
            f"cum. coverage = {cum_cov:.4f}, "
        )

    online_df = pd.DataFrame(
        {
            "step": np.arange(1, num_online_steps + 1),
            "mean_rating": online_mean_ratings,
            "mean_regret": online_mean_regrets,
            "distinct_items": step_distinct_items,
            "cum_coverage": step_cum_coverage,
        }
    )
    if save_path is None:
        online_df.to_csv(
            "online_latent_static_fixed_emb_metrics.csv",
            index=False
        )
    else:
        online_df.to_csv(
            save_path,
            index=False
        )
    print("Saved online metrics to online_latent_static_fixed_emb_metrics.csv")


def run_dynamic_simulation_fix(
    emb_path: str,
    save_path: str,
    device: str = "cuda",
    num_steps: int = 1000,
    users_per_step: int = 200,
    sigma_eps: float = 0.25,
    clip_min: float = 0.5,
    clip_max: float = 5.0,
    alpha: float = 1.0,
    affinity_rate: float = 0.05,
    boredom_lambda: float = 0.5,
    boredom_memory: int = 10,
    boredom_threshold: int = 3,
    similarity_threshold: float = 0.1,
    seed: int = 42,
    rating_threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Run online simulation based on following steps:
        1: Load lightgcn embeddings.
        2. At each step recommend items to a batch of users.
        3. Query env to calculate ratings for recommendation.
        4. Update embeddings based on Boredom and Affinity
        5. Calculate batch means.
    """
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    rating_df = pd.read_csv(r"data/ml-latest-small/ratings.csv")
    user_ids, users = pd.factorize(rating_df["userId"])
    item_ids, items = pd.factorize(rating_df["movieId"])
    rating_df["user_idx"] = user_ids
    rating_df["item_idx"] = item_ids

    embeddings = torch.load(emb_path, map_location=torch_device)
    user_emb_t: torch.Tensor = embeddings["user_emb"]
    item_emb_t: torch.Tensor = embeddings["item_emb"]

    num_users = user_emb_t.size(0)
    num_items = item_emb_t.size(0)

    rng = np.random.RandomState(seed)

    env = LatentDynamicEnv(
        num_users=num_users,
        num_items=num_items,
        user_emb=None,
        item_emb=None,
        ratings_df=rating_df,
        rng=rng,
        sigma_eps=sigma_eps,
        clip_min=clip_min,
        clip_max=clip_max,
        alpha=alpha,
        affinity_rate=affinity_rate,
        boredom_lambda=boredom_lambda,
        boredom_memory=boredom_memory,
        boredom_threshold=boredom_threshold,
        similarity_threshold=similarity_threshold,
    )

    online_mean_ratings = []
    online_num_interactions = []
    boredom_triggers_per_step = []
    mean_regrets = []
    distinct_items_per_step = []
    cum_coverage_per_step = []

    seen_items_per_user: dict[int, set[int]] = {u: set() for u in range(num_users)}

    global_recommended_items: set[int] = set()

    for t in range(1, num_steps + 1):
        batch_size = min(users_per_step, num_users)
        batch_users = rng.choice(num_users, size=batch_size, replace=False)

        step_ratings = []
        boredom_triggers_this_step = 0

        step_regrets = []
        step_items = []

        for u in batch_users:
            u = int(u)

            rec_item = recommend_one_item_for_user_from_emb(
                user_emb=user_emb_t,
                item_emb=item_emb_t,
                train_user_items=seen_items_per_user,
                user_idx=u,
                device=torch_device,
            )
            if rec_item is None:
                continue

            rec_item = int(rec_item)

            penalty = env._boredom_penalty(u, rec_item)
            if penalty > 0:
                boredom_triggers_this_step += 1

            r = env.interact(u, rec_item)
            step_ratings.append(r)

            seen_items_per_user[u].add(rec_item)

            oracle_r = oracle_mean_rating_for_user(env, u, num_items)
            step_regrets.append(oracle_r - r)

            step_items.append(rec_item)
            global_recommended_items.add(rec_item)

        mean_r = float(np.mean(step_ratings)) if step_ratings else np.nan
        online_mean_ratings.append(mean_r)
        online_num_interactions.append(len(step_ratings))
        boredom_triggers_per_step.append(boredom_triggers_this_step)

        mean_regret = float(np.mean(step_regrets)) if step_regrets else np.nan
        mean_regrets.append(mean_regret)

        distinct_cnt = len(set(step_items))
        distinct_items_per_step.append(distinct_cnt)

        cum_cov = len(global_recommended_items) / float(num_items)
        cum_coverage_per_step.append(cum_cov)

        print(
            f"[step {t:4d}] mean rating = {mean_r:.4f}  "
            f"mean regret = {mean_regret:.4f}  "
            f"(#interactions={len(step_ratings)}, "
            f"boredom_triggers={boredom_triggers_this_step}, "
            f"distinct_items={distinct_cnt}, "
            f"cum_coverage={cum_cov:.4f})"
        )

    result_df = pd.DataFrame(
        {
            "step": np.arange(1, num_steps + 1),
            "mean_rating": online_mean_ratings,
            "mean_regret": mean_regrets,
            "num_interactions": online_num_interactions,
            "boredom_triggers": boredom_triggers_per_step,
            "distinct_items": distinct_items_per_step,
            "cum_coverage": cum_coverage_per_step,
        }
    )

    result_df.to_csv(save_path, index=False)
    print(f"Saved dynamic online metrics to {save_path}")

    return result_df



def run_dynamic_simulation_retrain( 
    emb_path: str,
    save_path: str,
    device: str = "cuda",
    num_steps: int = 1000,
    users_per_step: int = 200,
    sigma_eps: float = 0.25,
    clip_min: float = 0.5,
    clip_max: float = 5.0,
    alpha: float = 1.0,
    affinity_rate: float = 0.05,
    boredom_lambda: float = 0.5,
    boredom_memory: int = 10,
    boredom_threshold: int = 3,
    similarity_threshold: float = 0.1,
    seed: int = 42,
    rating_threshold: float = 3.5,
    retrain_interval: int = 100,
) -> pd.DataFrame:
    """
    Run online simulation based on following steps:
        1: Load lightgcn embeddings.
        2. At each step recommend items to a batch of users.
        3. Query env to calculate ratings for recommendation.
        4. Update embeddings based on Boredom and Affinity
        5. Calculate batch means.
        6. Retrain the model by using all existing data every 100 iterations.
    """
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    rating_df = pd.read_csv(r"data/ml-latest-small/ratings.csv")
    user_ids, users = pd.factorize(rating_df["userId"])
    item_ids, items = pd.factorize(rating_df["movieId"])
    rating_df["user_idx"] = user_ids
    rating_df["item_idx"] = item_ids

    embeddings = torch.load(emb_path, map_location=torch_device)
    user_emb_t: torch.Tensor = embeddings["user_emb"]
    item_emb_t: torch.Tensor = embeddings["item_emb"]

    num_users = user_emb_t.size(0)
    num_items = item_emb_t.size(0)

    rng = np.random.RandomState(seed)

    env = LatentDynamicEnv(
        num_users=num_users,
        num_items=num_items,
        user_emb=None,
        item_emb=None,
        ratings_df=rating_df,
        rng=rng,
        sigma_eps=sigma_eps,
        clip_min=clip_min,
        clip_max=clip_max,
        alpha=alpha,
        affinity_rate=affinity_rate,
        boredom_lambda=boredom_lambda,
        boredom_memory=boredom_memory,
        boredom_threshold=boredom_threshold,
        similarity_threshold=similarity_threshold,
    )


    logged_interactions: list[dict] = []

    online_mean_ratings = []
    online_num_interactions = []
    boredom_triggers_per_step = []
    mean_regrets = []
    distinct_items_per_step = []
    cum_coverage_per_step = []

    seen_items_per_user: dict[int, set[int]] = {u: set() for u in range(num_users)}
    global_recommended_items: set[int] = set()


    for t in range(1, num_steps + 1):
        batch_size = min(users_per_step, num_users)
        batch_users = rng.choice(num_users, size=batch_size, replace=False)

        step_ratings = []
        boredom_triggers_this_step = 0
        step_regrets = []
        step_items = []

        for u in batch_users:
            u = int(u)

            rec_item = recommend_one_item_for_user_from_emb(
                user_emb=user_emb_t,
                item_emb=item_emb_t,
                train_user_items=seen_items_per_user,
                user_idx=u,
                device=torch_device,
            )
            if rec_item is None:
                continue

            rec_item = int(rec_item)

            penalty = env._boredom_penalty(u, rec_item)
            if penalty > 0:
                boredom_triggers_this_step += 1

            r = env.interact(u, rec_item)
            step_ratings.append(r)

            seen_items_per_user[u].add(rec_item)

            logged_interactions.append(
                {
                    "user_idx": u,
                    "item_idx": rec_item,
                    "rating": float(r),
                }
            )

            oracle_r = oracle_mean_rating_for_user(env, u, num_items)
            step_regrets.append(oracle_r - r)

            step_items.append(rec_item)
            global_recommended_items.add(rec_item)

        mean_r = float(np.mean(step_ratings)) if step_ratings else np.nan
        online_mean_ratings.append(mean_r)
        online_num_interactions.append(len(step_ratings))
        boredom_triggers_per_step.append(boredom_triggers_this_step)

        mean_regret = float(np.mean(step_regrets)) if step_regrets else np.nan
        mean_regrets.append(mean_regret)

        distinct_cnt = len(set(step_items))
        distinct_items_per_step.append(distinct_cnt)

        cum_cov = len(global_recommended_items) / float(num_items)
        cum_coverage_per_step.append(cum_cov)

        print(
            f"[step {t:4d}] mean rating = {mean_r:.4f}  "
            f"mean regret = {mean_regret:.4f}  "
            f"(#interactions={len(step_ratings)}, "
            f"boredom_triggers={boredom_triggers_this_step}, "
            f"distinct_items={distinct_cnt}, "
            f"cum_coverage={cum_cov:.4f})"
        )

        # Retrain the GNN model
        if (t % retrain_interval == 0) and len(logged_interactions) > 0 and t != num_steps:
            print(f"\n[retrain] step {t}: retraining model with offline + {len(logged_interactions)} online interactions")

            online_df = pd.DataFrame(logged_interactions)

            base_df = rating_df[["user_idx", "item_idx", "rating"]].copy()

            combined_df = pd.concat([base_df, online_df], ignore_index=True)

            user_emb_t, item_emb_t = retrain_model_from_df(
                combined_df=combined_df,
                num_users=num_users,
                num_items=num_items,
                device=torch_device,
            )

            print(f"[retrain] step {t}: model updated, continue simulation\n")
            # logged_interactions = []

    result_df = pd.DataFrame(
        {
            "step": np.arange(1, num_steps + 1),
            "mean_rating": online_mean_ratings,
            "mean_regret": mean_regrets,
            "num_interactions": online_num_interactions,
            "boredom_triggers": boredom_triggers_per_step,
            "distinct_items": distinct_items_per_step,
            "cum_coverage": cum_coverage_per_step,
        }
    )

    result_df.to_csv(save_path, index=False)
    print(f"Saved dynamic online metrics to {save_path}")

    return result_df
