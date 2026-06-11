"""
run_hidden_gcn.py
=================

Hidden Link Recovery Experiment — GCN (Model) pipeline.

What this script does
---------------------
1.  Loads the full DDA matrix.
2.  Hides a configurable fraction of positive links (hidden_ratio).
3.  Calls the ORIGINAL main.py training code exactly as-is, but using the
    modified DDA matrix that has hidden links removed from the graph.
4.  After all folds finish, evaluates whether the trained models can recover
    the hidden links.

Nothing in model.py, main.py, load_data.py, or utils.py is changed.
The GCN training procedure is IDENTICAL to running main.py directly.

Usage
-----
    python run_hidden_gcn.py -da Cdataset -ft Morgan -ct graph_ae \
           --hidden_ratio 0.10 --exp_name GCN_hidden

Arguments inherit ALL arguments from args.py, plus:
    --hidden_ratio   float   fraction of positives to hide  (default 0.10)

The script writes a hidden_link_report.txt alongside the normal result files.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from sklearn.model_selection import KFold

# ── Project imports (unchanged) ────────────────────────────────────────────
from model import Model
from load_data import load_dataset, remove_graph, generate_feat
from utils import (
    define_logging, get_metrics_auc, set_seed,
    plot_result_auc, plot_result_aupr, EarlyStopping, get_metrics,
)

from args import args
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ── Hidden-link extension (NEW — only addition) ────────────────────────────
from hidden_link_eval import hide_positive_links, evaluate_hidden_link_recovery


# ── Add --hidden_ratio to the already-parsed args namespace ────────────────
# args is already constructed in args.py; we just attach the new attribute.
import argparse as _ap
_extra = _ap.ArgumentParser(add_help=False)
_extra.add_argument('--hidden_ratio', default=0.10, type=float,
                    help='Fraction of positive DDA links to hide before training.')
_extra.add_argument('--threshold', default=0.5, type=float,
                    help='Score cutoff for counting a hidden link as recovered (default 0.5).')
_known, _ = _extra.parse_known_args()
args.hidden_ratio = _known.hidden_ratio
args.threshold    = _known.threshold


# ===========================================================================
# Training function — mirrors main.py train() exactly, with two additions:
#   (A) hide_positive_links() before building graphs/folds
#   (B) evaluate_hidden_link_recovery() after all folds complete
# ===========================================================================

def train():
    set_seed(args.seed)
    
     # إنشاء مجلد hidden_expr داخل مجلد الـ seed الحالي
    args.saved_path = os.path.join(
         args.saved_path,
         f"hidden_exprr_ratio_{args.hidden_ratio}"
    )

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    simplefilter(action='ignore', category=FutureWarning)
    logger = logging.getLogger('gcn_hidden_logger')
    logger.setLevel(logging.INFO)
    define_logging(args, logger)
    logger.info(args)
    logger.info('\n[Hidden-Link Experiment]  hidden_ratio = {:.2f}'.format(
        args.hidden_ratio))

    device = th.device('cuda:{}'.format(args.device_id)) \
        if args.device_id else th.device('cpu')
    args.device = device

    # ── Load full DDA matrix ────────────────────────────────────────────────
    full_df = pd.read_csv(
        '../data/{}/drug_dis.csv'.format(args.dataset), header=None
    ).values.astype('int64')

    # ── (A) Hide a fraction of positive links ──────────────────────────────
    # hidden_links : coordinates that are REMOVED from training entirely
    # visible_df   : the matrix the model will see during training
    hidden_links, visible_df = hide_positive_links(
        full_df, hidden_ratio=args.hidden_ratio, seed=args.seed
    )
    logger.info('Total positives: {}  |  Hidden: {}  |  Visible: {}'.format(
        int(full_df.sum()),
        len(hidden_links),
        int(visible_df.sum()),
    ))

    # ── Build data arrays from the VISIBLE (redacted) matrix ───────────────
    data = np.array([
        [i, j, visible_df[i, j]]
        for i in range(visible_df.shape[0])
        for j in range(visible_df.shape[1])
    ]).astype('int64')
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    assert len(data) == len(data_pos) + len(data_neg)

    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)
    fold        = 1
    pred_result = np.zeros(visible_df.shape)

    # Keep a reference to the last fold's trained model + graphs + features
    # for hidden-link evaluation (evaluated once, after all folds finish).
    last_model   = None
    last_g       = None
    last_g_llm   = None
    last_feature = None

    # ── k-Fold loop — IDENTICAL to main.py ─────────────────────────────────
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(
        kf.split(data_pos), kf.split(data_neg)
    ):
        logger.info('{}-Cross Validation: Fold {}'.format(args.nfold, fold))

        train_pos_id, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
        train_neg_id, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]
        train_pos_idx_t = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
        test_pos_idx_t  = [tuple(test_pos_id[:, 0]),  tuple(test_pos_id[:, 1])]
        train_neg_idx_t = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]
        test_neg_idx_t  = [tuple(test_neg_id[:, 0]),  tuple(test_neg_id[:, 1])]
        assert (len(test_pos_idx_t[0]) + len(test_neg_idx_t[0])
                + len(train_pos_idx_t[0]) + len(train_neg_idx_t[0])) == len(data)

        # Build graph — pass hidden_links so load_dataset excludes them
        g, g_llm = load_dataset(args, exclude_dr_di_edges=hidden_links)
        logger.info(g)

        # Remove test-fold edges (standard k-fold leakage prevention)
        g     = remove_graph(g,     test_pos_id).to(device)
        g_llm = remove_graph(g_llm, test_pos_id).to(device)

        feature = generate_feat(args, [g, g_llm])

        # Masks
        mask_label = np.ones(visible_df.shape)
        mask_label[test_pos_idx_t[0],  test_pos_idx_t[1]]  = 0
        mask_label[test_neg_idx_t[0],  test_neg_idx_t[1]]  = 0
        mask_test  = tuple(np.where(mask_label == 0))
        mask_train = tuple(np.where(mask_label == 1))

        logger.info('Train: {}  |  Test: {}  '
                    '(pos {}, neg {})'.format(
                        len(mask_train[0]), len(mask_test[0]),
                        len(train_pos_idx_t[0]), len(test_pos_idx_t[0])))

        label = th.tensor(visible_df).float().to(device)

        # Model
        if args.concatenate_type in ['none', 'as_node']:
            model = Model(args=args, etypes=g.etypes, ntypes=g.ntypes,
                          in_feats=[feature['drug'].shape[1],
                                    feature['disease'].shape[1]])
        else:
            model = Model(args=args, etypes=g.etypes, ntypes=g.ntypes,
                          in_feats=[feature['drug'].shape[1],
                                    feature['disease'].shape[1],
                                    feature['drug_LLM'].shape[1],
                                    feature['disease_LLM'].shape[1]])
        model.to(device)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
        optim_scheduler = th.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.1 * args.learning_rate,
            max_lr=args.learning_rate,
            gamma=0.995, step_size_up=20,
            mode='exp_range', cycle_momentum=False,
        )
        criterion = th.nn.BCEWithLogitsLoss(
            pos_weight=th.tensor(len(train_neg_idx_t[0]) / len(train_pos_idx_t[0]))
        )
        logger.info('Loss pos weight: {:.3f}'.format(
            len(train_neg_idx_t[0]) / len(train_pos_idx_t[0])))

        # Training loop — identical to main.py
        for epoch in range(1, args.epoch + 1):
            model.train()
            score = model([g, g_llm], feature)
            pred  = th.sigmoid(score)
            loss  = criterion(
                score[mask_train].cpu().flatten(),
                label[mask_train].cpu().flatten(),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optim_scheduler.step()
            model.eval()
            AUC_, _ = get_metrics_auc(
                label[mask_train].cpu().detach().numpy(),
                pred[mask_train].cpu().detach().numpy(),
            )
            if epoch % 50 == 0:
                AUC, AUPR = get_metrics_auc(
                    label[mask_test].cpu().detach().numpy(),
                    pred[mask_test].cpu().detach().numpy(),
                )
                logger.info(
                    'Epoch {} Loss: {:.3f}; Train AUC {:.3f}; '
                    'AUC {:.3f}; AUPR: {:.3f}'.format(
                        epoch, loss.item(), AUC_, AUC, AUPR))

        model.eval()
        pred_np = th.sigmoid(model([g, g_llm], feature)).cpu().detach().numpy()
        test_pos_arr = np.array(test_pos_idx_t)
        test_neg_arr = np.array(test_neg_idx_t)
        pred_result[test_pos_arr[0], test_pos_arr[1]] = \
            pred_np[test_pos_arr[0], test_pos_arr[1]]
        pred_result[test_neg_arr[0], test_neg_arr[1]] = \
            pred_np[test_neg_arr[0], test_neg_arr[1]]

        th.save(model.state_dict(),
                os.path.join(args.saved_path, 'model_fold_{}.pth'.format(fold)))

        # Retain final fold artefacts for hidden-link evaluation
        last_model   = model
        last_g       = g
        last_g_llm   = g_llm
        last_feature = feature

        fold += 1

    
# ── Standard overall metrics (memory-safe version) ──────────────────────
    label_all = th.tensor(visible_df).float()

    y_true = label_all.numpy().flatten()
    y_score = pred_result.flatten()

   # Ranking metrics
    AUC = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    # Binary metrics at threshold 0.5
    y_pred = (y_score >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info(
        'Overall (standard CV): AUC {:.4f}; AUPR {:.4f}; '
        'Acc {:.4f}; F1 {:.4f}; Prec {:.4f}; Recall {:.4f}'.format(
            AUC, aupr, acc, f1, pre, rec))

    pd.DataFrame(pred_result).to_csv(
        os.path.join(args.saved_path, 'result.csv'), index=False, header=False)
    plot_result_auc(args,  label_all.numpy().flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, label_all.numpy().flatten(), pred_result.flatten(), aupr)

    # ── (B) Hidden-link recovery evaluation ────────────────────────────────
    # Uses the model from the last fold. Hidden links were completely absent
    # from the graph during all folds; the model has never seen them.
    report = evaluate_hidden_link_recovery(
        model        = last_model,
        g            = last_g,
        g_llm        = last_g_llm,
        feature      = last_feature,
        hidden_links = hidden_links,
        device       = device,
        logger       = logger,
        threshold    = args.threshold,
    )

    # ── Save hidden-link report ─────────────────────────────────────────────
    report_path = os.path.join(args.saved_path, 'hidden_link_report.txt')
    with open(report_path, 'w') as fh:
        fh.write('=== Hidden Link Recovery Report (GCN) ===\n')
        fh.write('hidden_ratio    : {:.4f}\n'.format(args.hidden_ratio))
        fh.write('threshold       : {:.4f}\n'.format(args.threshold))
        fh.write('n_hidden        : {:d}\n'.format(report['n_hidden']))
        fh.write('n_recovered     : {:d}\n'.format(report['n_recovered']))
        fh.write('Recovery Rate   : {:.4f}  ({:.1f}%)\n'.format(
            report['recovery_rate'], report['recovery_rate'] * 100))
        fh.write('Mean Score      : {:.4f}\n'.format(report['mean_score_hidden']))
        fh.write('Median Score    : {:.4f}\n'.format(report['median_score_hidden']))
        fh.write('\n--- Standard CV Metrics ---\n')
        fh.write('AUC     : {:.4f}\n'.format(AUC))
        fh.write('AUPR    : {:.4f}\n'.format(aupr))
        fh.write('Acc     : {:.4f}\n'.format(acc))
        fh.write('F1      : {:.4f}\n'.format(f1))
        fh.write('Prec    : {:.4f}\n'.format(pre))
        fh.write('Recall  : {:.4f}\n'.format(rec))
    logger.info('Hidden-link report saved to: {}'.format(report_path))


if __name__ == '__main__':
    train()