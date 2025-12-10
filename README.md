Below is a brief description of each Python file in our repository.
1. data_preprocessing.py — Loads and preprocesses the MovieLens dataset and builds graph/user–item structures.

2. eval.py — Simulate the online interactions between users and recommender under all environments.

3. lightgcn.py — Implements the LightGCN graph-based collaborative filtering model.

4. gat.py — Implements the Graph Attention Network recommender model.

5. graphsage.py — Implements the GraphSAGE-based recommender model.

6. latent_static_env.py — RecLab-style latent-static simulator generating ratings based on fixed user/item embeddings.

7. latent_dynamic_env.py — RecLab-style latent-dynamic simulator with user-state updates (affinity + boredom).


Below is a brief description of each folder in our repository.

1. embeddings/ — Stores pretrained user and item node embeddings for three GNN models.

2. figures/ — Contains three result plots.

3. notebooks/ — Jupyter notebooks for testing abd debugging.