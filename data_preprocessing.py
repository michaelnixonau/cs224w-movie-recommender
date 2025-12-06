import numpy as np
import pandas as pd
import torch
import os

def build_dataset_static(ratings_path, device):
    ext = os.path.splitext(ratings_path)[1].lower()
    if ext == ".csv":
        ratings = pd.read_csv(ratings_path)
    elif ext == ".dat":
        ratings = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            header=None,
            names=["userId", "movieId", "rating", "timestamp"]
        )
    else: 
        raise ValueError(f'Wrong file format: {ext}')

    user_ids, users = pd.factorize(ratings['userId'])
    item_ids, items = pd.factorize(ratings['movieId'])

    num_users = users.size
    num_items = items.size
    num_nodes = num_users + num_items

    rng = np.random.RandomState(42)

    # Train-Test Set Split by Strategy of per-user leave-one-out
    positive_edges = {}  # All positive edges
    for u, i in zip(user_ids, item_ids):
        positive_edges.setdefault(u, set()).add(i)

    test_pairs = []  # Test Set
    train_user_items = {}  # Train Set Raw data

    for u, item_set in positive_edges.items():
        # If number of ratings for one user less than 2
        # Put all of them in training set
        if len(item_set) < 2:
            train_user_items[u] = set(item_set)
        else:
            item_list = list(item_set)
            i_test = rng.choice(item_list)  # Randomly choose one edge as test case
            test_pairs.append((u, i_test))
            item_list.remove(i_test)
            train_user_items[u] = set(item_list)

    # Reconstruct the training graph
    train_u, train_i = [], []
    for u, S in train_user_items.items():
        for i in S:
            train_u.append(u)
            train_i.append(i)
    train_u = np.array(train_u)
    train_i = np.array(train_i)

    src = torch.tensor(train_u, dtype=torch.long)
    dst = torch.tensor(train_i + num_users, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)  # Each column represents an edge
    # Add reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)

    # Train Set
    train_set = {u: set() for u in range(num_users)}
    for u, i in zip(train_u, train_i):
        train_set[u].add(i)

    return (
        num_users, num_items, num_nodes,
        edge_index, train_set,
        train_user_items, test_pairs
    )



def build_dataset_dynamic(ratings_path, device):
    ext = os.path.splitext(ratings_path)[1].lower()
    if ext == ".csv":
        ratings = pd.read_csv(ratings_path)
    elif ext == ".dat":
        ratings = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            header=None,
            names=["userId", "movieId", "rating", "timestamp"],
        )


    user_ids, users = pd.factorize(ratings["userId"])
    item_ids, items = pd.factorize(ratings["movieId"])

    num_users = users.size
    num_items = items.size
    num_nodes = num_users + num_items

    ratings["user_idx"] = user_ids.astype(np.int64)
    ratings["item_idx"] = item_ids.astype(np.int64)

    ratings = ratings.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)
    ratings["t"] = ratings.groupby("user_idx").cumcount()

    rng = np.random.RandomState(42)

    # Train-Test Set Split by Strategy of per-user leave-one-out
    positive_edges = {}  # All positive edges
    for u, i in zip(user_ids, item_ids):
        positive_edges.setdefault(u, set()).add(i)

    test_pairs = []          # Test set
    train_user_items = {}    # Train set (per user: set of items)

    for u, group in ratings.groupby("user_idx"):
        items = group["item_idx"].to_numpy(dtype=np.int64)

        if len(items) < 2:
            train_user_items[int(u)] = set(items.tolist())
        else:
            i_test = int(items[-1])
            test_pairs.append((int(u), i_test))

            train_items = items[:-1]
            train_user_items[int(u)] = set(train_items.tolist())

    # Reconstruct the training graph
    train_u, train_i = [], []
    for u, S in train_user_items.items():
        for i in S:
            train_u.append(u)
            train_i.append(i)

    train_u = np.array(train_u, dtype=np.int64)
    train_i = np.array(train_i, dtype=np.int64)

    src = torch.tensor(train_u, dtype=torch.long)
    dst = torch.tensor(train_i + num_users, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    # Add reverse edges
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to(device)

    # Train set as dict[user] -> set(items)
    train_set = {u: set() for u in range(num_users)}
    for u, i in zip(train_u, train_i):
        train_set[u].add(i)

    return (
        num_users,
        num_items,
        num_nodes,
        edge_index,
        train_set,
        train_user_items,
        test_pairs,
        ratings,  # ratings_df with user_idx, item_idx, rating, timestamp, t
    )


def sample_triples(batch_size, train_set, num_users, num_items, device, rng):
    # Return batch of positive samples ui and negative samples uj
    us, is_, js = [], [], []
    users = [u for u in range(num_users) if len(train_set[u]) > 0]

    for _ in range(batch_size):
        # Randomly sample a user u from train set
        u = rng.choice(users)

        # Positive sample for user u
        i = rng.choice(list(train_set[u]))

        # Negative sample for user u
        j = rng.randint(0, num_items)
        while j in train_set[u]:
            j = rng.randint(0, num_items)

        us.append(u)
        is_.append(i)
        js.append(j)

    us = torch.tensor(us, dtype=torch.long, device=device)
    is_ = torch.tensor(is_, dtype=torch.long, device=device)
    js = torch.tensor(js, dtype=torch.long, device=device)
    return us, is_, js