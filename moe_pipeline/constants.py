N_DESC = 200    # number of RDKit descriptor columns in feature matrix
K_OOF  = 5     # inner out-of-fold folds for honest expert predictions

NON_FEAT_COLS = ["smiles", "pIC50", "target", "scaffold", "scaffold_idx"]

HP = dict(
    seed=87,
    n_folds=4,

    # Gate architecture & training (fixed across the search)
    gate_hidden_sizes=[100, 80],
    gate_lr=1e-3,
    gate_epochs=300,
    gate_oracle_reg=0.01,

    # Gate L2 regularization grid
    l2_regs=[0.0, 0.001, 0.01, 0.1],

    # Oracle regularization grid
    # Cross-entropy loss pushing routing weights toward the per-sample best expert.
    oracle_regs=[0.0, 0.01, 0.1],

    # Gate loss configuration grid
    # entropy_reg > 0: bonus for spreading weight across experts (near-uniform routing).
    # load_balancing = True: Switch-Transformer penalty for routing collapse.
    loss_configs=[
        dict(entropy_reg=0.0,  load_balancing=False),   # baseline
        dict(entropy_reg=1e-4, load_balancing=False),   # very mild entropy
        dict(entropy_reg=1e-3, load_balancing=False),   # mild entropy
        dict(entropy_reg=1e-2, load_balancing=False),   # moderate entropy
        dict(entropy_reg=0.1,  load_balancing=False),   # strong: near-uniform routing
        dict(entropy_reg=0.0,  load_balancing=True),    # load balancing instead
    ],

    # Gate routing scheme grid
    # top_k=None: soft routing (all experts contribute).
    # top_k=1/2/3: sparse routing (only top-k experts used per sample).
    routing_schemes=[
        dict(top_k=None),
        dict(top_k=1),
        dict(top_k=2),
        dict(top_k=3),
    ],

    # Expert model configurations — 4 configs = corners of (tabular-reg) x (GNN-reg)
    expert_configs=[
        # A — baseline: matches _make_experts() defaults
        dict(n_estimators=200, learning_rate=0.05, lgbm_reg_lambda=0.0,
             rf_max_features=1.0,    dmpnn_dropout=0.0, gin_num_layers=4),
        # B — stronger tabular: more trees + lower lr, LightGBM L2, RF feature diversity
        dict(n_estimators=500, learning_rate=0.02, lgbm_reg_lambda=1.0,
             rf_max_features="sqrt", dmpnn_dropout=0.0, gin_num_layers=4),
        # C — GNN regularization: DMPNN dropout, shallower GIN (less over-smoothing)
        dict(n_estimators=200, learning_rate=0.05, lgbm_reg_lambda=0.0,
             rf_max_features=1.0,    dmpnn_dropout=0.2, gin_num_layers=3),
        # D — all tuned: combines B tabular + C GNN regularization
        dict(n_estimators=500, learning_rate=0.02, lgbm_reg_lambda=1.0,
             rf_max_features="sqrt", dmpnn_dropout=0.2, gin_num_layers=3),
    ],
)
