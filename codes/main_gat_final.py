

import os
import logging
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from sklearn.model_selection import KFold, train_test_split

from model_gat_final import ModelGAT
from load_data import load_dataset, remove_graph, generate_feat
from utils import (
    define_logging, get_metrics_auc, set_seed,
    plot_result_auc, plot_result_aupr, EarlyStopping, get_metrics,
)
from args_gat_final import args   # separate args file for GAT


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build model given feature dict
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args, g, feature: dict) -> ModelGAT:
    """Instantiate ModelGAT with the correct input-feature dimensions."""
    if args.concatenate_type in ['none', 'as_node']:
        in_feats = [
            feature['drug'].shape[1],
            feature['disease'].shape[1],
        ]
    else:
        in_feats = [
            feature['drug'].shape[1],
            feature['disease'].shape[1],
            feature['drug_LLM'].shape[1],
            feature['disease_LLM'].shape[1],
        ]

    model = ModelGAT(
        args=args,
        etypes=g.etypes,
        ntypes=g.ntypes,
        in_feats=in_feats,
    )
    return model.to(args.device)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: one training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, optimizer, criterion, g, g_llm, feature,
    label, mask_train, clip_grad: float = 1.0,
):
    """
    Run one forward + backward pass over the training mask.

    Returns
    -------
    loss_val : float
    train_auc : float
    """
    model.train()
    score = model([g, g_llm], feature)           # [N_drug, N_disease]
    pred  = th.sigmoid(score)

    # Compute loss only on training pairs
    loss = criterion(
    score[mask_train].flatten(),
    label[mask_train].flatten(),
)

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping — important for GAT which can have large attention
    # gradients early in training.
    if clip_grad > 0:
        th.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    optimizer.step()

    with th.no_grad():
       auc, _ = get_metrics_auc(
    label[mask_train].detach().cpu().numpy(),
    pred[mask_train].detach().cpu().numpy(),
)
    return loss.item(), auc


# ─────────────────────────────────────────────────────────────────────────────
# Helper: evaluate on a mask (no gradient)
# ─────────────────────────────────────────────────────────────────────────────
@th.no_grad()
def evaluate(model, g, g_llm, feature, label, mask):
    """Return AUC and AUPR for the given mask."""

    model.eval()

    score = model([g, g_llm], feature)

    pred = th.sigmoid(score)

    auc, aupr = get_metrics_auc(
        label[mask].detach().cpu().numpy(),
        pred[mask].detach().cpu().numpy(),
    )

    return auc, aupr, pred

# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train():
    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(args.seed)

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    simplefilter(action='ignore', category=FutureWarning)

    logger = logging.getLogger('gat_logger')
    logger.setLevel(logging.INFO)
    define_logging(args, logger)
    logger.info(args)
    logger.info('\n' + '='*70)
    logger.info('GAT MODEL — Drug-Disease Association Prediction')
    logger.info('='*70)
    logger.info(
        '\n[LEAKAGE WARNING] Similarity matrices (dr_sim, di_sim) are computed\n'
        'over ALL nodes before the train/test split.  When these matrices are\n'
        'derived from the DDA label matrix (Gaussian/Jaccard kernel), test\n'
        'association information leaks into graph construction.\n'
        'Set --leak_safe_mode to True to recompute per-fold (slower).\n'
        'For chemical/sequence similarities (Morgan, ChemBERTa, BERT) this\n'
        'risk is lower because they are label-independent.\n'
    )

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device_id:
        logger.info('Training on GPU {}'.format(args.device_id))
        device = th.device('cuda:{}'.format(args.device_id))
    else:
        logger.info('Training on CPU')
        device = th.device('cpu')
    args.device = device

    # ── Load DDA matrix for k-fold splitting ──────────────────────────────────
    df = pd.read_csv(
        '../data/{}/drug_dis.csv'.format(args.dataset), header=None
    ).values
    data     = np.array([
        [i, j, df[i, j]]
        for i in range(df.shape[0])
        for j in range(df.shape[1])
    ]).astype('int64')
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    assert len(data) == len(data_pos) + len(data_neg)

    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)

    fold        = 1
    pred_result = np.zeros(df.shape)



    # ── k-Fold loop ───────────────────────────────────────────────────────────
    for (tr_pos_idx, te_pos_idx), (tr_neg_idx, te_neg_idx) in zip(
        kf.split(data_pos), kf.split(data_neg)
    ):
        logger.info('\n' + '-'*60)
        logger.info('{}-Fold CV  |  Fold {}'.format(args.nfold, fold))
        logger.info('-'*60)

        # ── Index arrays for this fold ─────────────────────────────────────
        train_pos_id, test_pos_id = data_pos[tr_pos_idx], data_pos[te_pos_idx]
        train_neg_id, test_neg_id = data_neg[tr_neg_idx], data_neg[te_neg_idx]

        # tuple-based masks for fancy indexing into [N_dr, N_di] tensors
        test_pos_idx_t  = [tuple(test_pos_id[:, 0]),  tuple(test_pos_id[:, 1])]
        test_neg_idx_t  = [tuple(test_neg_id[:, 0]),  tuple(test_neg_id[:, 1])]
        train_pos_idx_t = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
        train_neg_idx_t = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]

        assert (
            len(test_pos_idx_t[0]) + len(test_neg_idx_t[0])
            + len(train_pos_idx_t[0]) + len(train_neg_idx_t[0])
        ) == len(data)

        # ── Build graphs ───────────────────────────────────────────────────
        g, g_llm = load_dataset(args)
        logger.info('Graph: {}'.format(g))

        # CRITICAL: remove test-set drug-disease edges from BOTH graphs
        # before any forward pass.  This is the minimum required step to
        # prevent direct edge leakage.
        g     = remove_graph(g,     test_pos_id).to(device)
        g_llm = remove_graph(g_llm, test_pos_id).to(device)

        # ── Generate node features ─────────────────────────────────────────
        feature = generate_feat(args, [g, g_llm])
        # Move features to device
        feature = {k: v.to(device) if isinstance(v, th.Tensor) else v
                   for k, v in feature.items()}

        # ── Mask computation ───────────────────────────────────────────────
        # mask_label == 0  → held-out test pair
        # mask_label == 1  → available training pair
        mask_label = np.ones(df.shape, dtype=int)
        mask_label[test_pos_idx_t[0],  test_pos_idx_t[1]]  = 0
        mask_label[test_neg_idx_t[0],  test_neg_idx_t[1]]  = 0
        mask_test  = tuple(np.where(mask_label == 0))
        mask_train_full = tuple(np.where(mask_label == 1))

        # ── Validation split (10 % of training pairs) ─────────────────────
        # We split the flat indices of training pairs into train / val so
        # that early stopping is evaluated on held-out data within the fold.
        # This prevents overfitting to the test set via patience tuning.
        n_train = len(mask_train_full[0])
        train_flat, val_flat = train_test_split(
            np.arange(n_train),
            test_size=0.1,
            random_state=args.seed,
        )
        mask_train = (
            tuple(mask_train_full[0][train_flat]),
            tuple(mask_train_full[1][train_flat]),
        )
        mask_val = (
            tuple(mask_train_full[0][val_flat]),
            tuple(mask_train_full[1][val_flat]),
        )

        logger.info(
            'Train pairs: {}  |  Val pairs: {}  |  Test pairs: {}'.format(
                len(mask_train[0]), len(mask_val[0]), len(mask_test[0])
            )
        )

        label = th.tensor(df).float().to(device)

        # ── Build model ────────────────────────────────────────────────────
        model     = build_model(args, g, feature)
        n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Model parameters: {:,}'.format(n_params))

        optimizer = th.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = th.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.1 * args.learning_rate,
            max_lr=args.learning_rate,
            gamma=0.995,
            step_size_up=20,
            mode='exp_range',
            cycle_momentum=False,
        )

        # Positive weight for imbalanced labels
        pos_weight = len(train_neg_idx_t[0]) / max(len(train_pos_idx_t[0]), 1)
        criterion  = th.nn.BCEWithLogitsLoss(
            pos_weight=th.tensor(pos_weight)
        )
        logger.info('BCE pos_weight: {:.3f}'.format(pos_weight))

        # EarlyStopping monitors VALIDATION AUC (higher = better)
        best_val_auc   = -1.0
        patience_count = 0
        best_ckpt_path = os.path.join(
            args.saved_path, 'best_model_fold_{}.pth'.format(fold)
        )

        # ── Training loop ──────────────────────────────────────────────────
        for epoch in range(1, args.epoch + 1):
            loss_val, train_auc = train_one_epoch(
                model, optimizer, criterion,
                g, g_llm, feature,
                label, mask_train,
                clip_grad=args.clip_grad,
            )
            scheduler.step()

            # Evaluate on validation set every epoch for early stopping
            val_auc, val_aupr, _ = evaluate(
                model, g, g_llm, feature, label, mask_val
            )

            # Save best checkpoint
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_count = 0
                th.save(model.state_dict(), best_ckpt_path)
            else:
                patience_count += 1

            # Verbose logging every 50 epochs
            if epoch % 50 == 0:
                test_auc, test_aupr, _ = evaluate(
                    model, g, g_llm, feature, label, mask_test
                )
                logger.info(
                    'Epoch {:4d}  '
                    'Loss: {:.4f}  '
                    'TrainAUC: {:.4f}  '
                    'ValAUC: {:.4f}  '
                    'TestAUC: {:.4f}  '
                    'TestAUPR: {:.4f}  '
                    '(patience {}/{})'.format(
                        epoch, loss_val,
                        train_auc, val_auc,
                        test_auc, test_aupr,
                        patience_count, args.patience,
                    )
                )

            # Early stopping check
            # if patience_count >= args.patience:
            #     logger.info(
            #         'Early stopping at epoch {} '
            #         '(best val AUC: {:.4f})'.format(epoch, best_val_auc)
            #     )
            #     break

        # ── Load best checkpoint and compute test performance ──────────────
        model.load_state_dict(th.load(best_ckpt_path, map_location=device))
        model.eval()

        _, _, pred_tensor = evaluate(model, g, g_llm, feature, label, mask_test)
        pred_np = pred_tensor.cpu().detach().numpy()

        # Store fold predictions
        test_pos_idx_arr = np.array(test_pos_idx_t)
        test_neg_idx_arr = np.array(test_neg_idx_t)
        pred_result[test_pos_idx_arr[0], test_pos_idx_arr[1]] = \
            pred_np[test_pos_idx_arr[0], test_pos_idx_arr[1]]
        pred_result[test_neg_idx_arr[0], test_neg_idx_arr[1]] = \
            pred_np[test_neg_idx_arr[0], test_neg_idx_arr[1]]

        fold_auc, fold_aupr, _, _, _, _, _ = get_metrics(
            label[mask_test].cpu().numpy(),
            pred_np[mask_test[0], mask_test[1]].flatten(),
        )
        logger.info(
            'Fold {} FINAL  AUC: {:.4f}  AUPR: {:.4f}'.format(
                fold, fold_auc, fold_aupr
            )
        )

        fold += 1
   
    print("df shape =", df.shape)
    print("pred_result shape =", pred_result.shape)
    print("flatten size =", pred_result.size)

    # ── Overall evaluation ────────────────────────────────────────────────────
    label_np = th.tensor(df).float().numpy()
    AUC, aupr, acc, f1, pre, rec, spec = get_metrics(
        label_np.flatten(), pred_result.flatten()
    )
    logger.info('\n' + '='*60)
    logger.info(
        'OVERALL  AUC: {:.4f}  AUPR: {:.4f}  '
        'Acc: {:.4f}  F1: {:.4f}  '
        'Prec: {:.4f}  Recall: {:.4f}'.format(
            AUC, aupr, acc, f1, pre, rec
        )
    )
    logger.info('='*60)

    # Save predictions and plots
    pd.DataFrame(pred_result).to_csv(
        os.path.join(args.saved_path, 'result_gat.csv'),
        index=False, header=False,
    )
    plot_result_auc(args,  label_np.flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, label_np.flatten(), pred_result.flatten(), aupr)

    logger.info('\nDone.  Results saved to: {}'.format(args.saved_path))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    train()