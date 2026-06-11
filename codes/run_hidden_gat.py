"""
run_hidden_gat.py
=================

Hidden Link Recovery Experiment — GAT (ModelGAT) pipeline.

What this script does
---------------------
1.  Loads the full DDA matrix.
2.  Hides a configurable fraction of positive links (hidden_ratio).
3.  Calls the ORIGINAL main_gat_final.py training code exactly as-is,
    but uses the modified DDA matrix that has hidden links removed.
4.  After all folds finish, evaluates whether the trained models can recover
    the hidden links.

Nothing in model_gat_final.py, main_gat_final.py, load_data.py, or
utils.py is changed.  The GAT training procedure is IDENTICAL to running
main_gat_final.py directly.

Usage
-----
    python run_hidden_gat.py -da Cdataset -ft Morgan -ct graph_ae \
           --hidden_ratio 0.10 --exp_name GAT_hidden

Arguments inherit ALL arguments from args_gat_final.py, plus:
    --hidden_ratio   float   fraction of positives to hide  (default 0.10)
"""

import os
import logging
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from sklearn.model_selection import KFold, train_test_split

# ── Project imports (unchanged) ────────────────────────────────────────────
from model_gat_final import ModelGAT
from load_data import load_dataset, remove_graph, generate_feat
from utils import (
    define_logging, get_metrics_auc, set_seed,
    plot_result_auc, plot_result_aupr, EarlyStopping, get_metrics,
)
from args_gat_final import args

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

# ── Reuse the helpers already defined in main_gat_final ────────────────────
from main_gat_final import build_model, train_one_epoch, evaluate

# ── Add --hidden_ratio to the already-parsed args namespace ────────────────
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
# Training function — mirrors main_gat_final.py train() exactly,
# with two additions:
#   (A) hide_positive_links() before building graphs/folds
#   (B) evaluate_hidden_link_recovery() after all folds complete
# ===========================================================================

def train():
    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(args.seed)


    # إنشاء مجلد hidden_expr داخل مجلد الـ seed الحالي
    args.saved_path = os.path.join(
         args.saved_path,
        f"hidden_exprr_ratio_{args.hidden_ratio}"
    )

    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)

    simplefilter(action='ignore', category=FutureWarning)

    logger = logging.getLogger('gat_hidden_logger')
    logger.setLevel(logging.INFO)
    define_logging(args, logger)
    logger.info(args)
    logger.info('\n[Hidden-Link Experiment]  hidden_ratio = {:.2f}'.format(
        args.hidden_ratio))

    # ── Device ────────────────────────────────────────────────────────────────
    device = th.device('cuda:{}'.format(args.device_id)) \
        if args.device_id else th.device('cpu')
    args.device = device

    # ── Load full DDA matrix ───────────────────────────────────────────────
    full_df = pd.read_csv(
        '../data/{}/drug_dis.csv'.format(args.dataset), header=None
    ).values.astype('int64')

    # ── (A) Hide a fraction of positive links ──────────────────────────────
    hidden_links, visible_df = hide_positive_links(
        full_df, hidden_ratio=args.hidden_ratio, seed=args.seed
    )
    logger.info('Total positives: {}  |  Hidden: {}  |  Visible: {}'.format(
        int(full_df.sum()), len(hidden_links), int(visible_df.sum())))

    # ── Build data arrays from visible matrix ─────────────────────────────
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

    # Retain last fold artefacts for hidden-link evaluation
    last_model   = None
    last_g       = None
    last_g_llm   = None
    last_feature = None

    # ── k-Fold loop — IDENTICAL to main_gat_final.py ──────────────────────
    for (tr_pos_idx, te_pos_idx), (tr_neg_idx, te_neg_idx) in zip(
        kf.split(data_pos), kf.split(data_neg)
    ):
        logger.info('\n' + '-'*60)
        logger.info('{}-Fold CV  |  Fold {}'.format(args.nfold, fold))
        logger.info('-'*60)

        train_pos_id, test_pos_id = data_pos[tr_pos_idx], data_pos[te_pos_idx]
        train_neg_id, test_neg_id = data_neg[tr_neg_idx], data_neg[te_neg_idx]

        test_pos_idx_t  = [tuple(test_pos_id[:, 0]),  tuple(test_pos_id[:, 1])]
        test_neg_idx_t  = [tuple(test_neg_id[:, 0]),  tuple(test_neg_id[:, 1])]
        train_pos_idx_t = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
        train_neg_idx_t = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]

        assert (
            len(test_pos_idx_t[0]) + len(test_neg_idx_t[0])
            + len(train_pos_idx_t[0]) + len(train_neg_idx_t[0])
        ) == len(data)

        # Build graphs — exclude hidden links from graph construction
        g, g_llm = load_dataset(args, exclude_dr_di_edges=hidden_links)
        logger.info('Graph: {}'.format(g))

        g     = remove_graph(g,     test_pos_id).to(device)
        g_llm = remove_graph(g_llm, test_pos_id).to(device)

        feature = generate_feat(args, [g, g_llm])
        feature = {k: v.to(device) if isinstance(v, th.Tensor) else v
                   for k, v in feature.items()}

        # Masks
        mask_label = np.ones(visible_df.shape, dtype=int)
        mask_label[test_pos_idx_t[0], test_pos_idx_t[1]] = 0
        mask_label[test_neg_idx_t[0], test_neg_idx_t[1]] = 0
        mask_test       = tuple(np.where(mask_label == 0))
        mask_train_full = tuple(np.where(mask_label == 1))

        # Validation split (10 % of training) — from main_gat_final.py
        n_train = len(mask_train_full[0])
        train_flat, val_flat = train_test_split(
            np.arange(n_train), test_size=0.1, random_state=args.seed
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
            'Train: {}  |  Val: {}  |  Test: {}'.format(
                len(mask_train[0]), len(mask_val[0]), len(mask_test[0])))

        label = th.tensor(visible_df).float().to(device)

        # Model, optimiser, scheduler — identical to main_gat_final.py
        model      = build_model(args, g, feature)
        n_params   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Model parameters: {:,}'.format(n_params))

        optimizer  = th.optim.Adam(model.parameters(),
                                   lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
        scheduler  = th.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.1 * args.learning_rate,
            max_lr=args.learning_rate,
            gamma=0.995, step_size_up=20,
            mode='exp_range', cycle_momentum=False,
        )
        pos_weight = len(train_neg_idx_t[0]) / max(len(train_pos_idx_t[0]), 1)
        criterion  = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(pos_weight))
        logger.info('BCE pos_weight: {:.3f}'.format(pos_weight))

        best_val_auc   = -1.0
        patience_count = 0
        best_ckpt_path = os.path.join(
            args.saved_path, 'best_model_fold_{}.pth'.format(fold))

        # Training loop — identical to main_gat_final.py
        for epoch in range(1, args.epoch + 1):
            loss_val, train_auc = train_one_epoch(
                model, optimizer, criterion,
                g, g_llm, feature, label, mask_train,
                clip_grad=args.clip_grad,
            )
            scheduler.step()

            val_auc, val_aupr, _ = evaluate(
                model, g, g_llm, feature, label, mask_val)

            if val_auc > best_val_auc:
                best_val_auc   = val_auc
                patience_count = 0
                th.save(model.state_dict(), best_ckpt_path)
            else:
                patience_count += 1

            if epoch % 50 == 0:
                test_auc, test_aupr, _ = evaluate(
                    model, g, g_llm, feature, label, mask_test)
                logger.info(
                    'Epoch {:4d}  Loss: {:.4f}  TrainAUC: {:.4f}  '
                    'ValAUC: {:.4f}  TestAUC: {:.4f}  '
                    'TestAUPR: {:.4f}  (patience {}/{})'.format(
                        epoch, loss_val, train_auc, val_auc,
                        test_auc, test_aupr,
                        patience_count, args.patience))

        # Load best checkpoint
        model.load_state_dict(th.load(best_ckpt_path, map_location=device))
        model.eval()

        _, _, pred_tensor = evaluate(model, g, g_llm, feature, label, mask_test)
        pred_np = pred_tensor.cpu().detach().numpy()

        test_pos_arr = np.array(test_pos_idx_t)
        test_neg_arr = np.array(test_neg_idx_t)
        pred_result[test_pos_arr[0], test_pos_arr[1]] = \
            pred_np[test_pos_arr[0], test_pos_arr[1]]
        pred_result[test_neg_arr[0], test_neg_arr[1]] = \
            pred_np[test_neg_arr[0], test_neg_arr[1]]

        fold_auc, fold_aupr, _, _, _, _, _ = get_metrics(
            label[mask_test].cpu().numpy(),
            pred_np[mask_test[0], mask_test[1]].flatten(),
        )
        logger.info('Fold {} FINAL  AUC: {:.4f}  AUPR: {:.4f}'.format(
            fold, fold_auc, fold_aupr))

        # Retain artefacts
        last_model   = model
        last_g       = g
        last_g_llm   = g_llm
        last_feature = feature

        fold += 1

    # ── Standard overall metrics (same as main_gat_final.py) ──────────────
    label_np = th.tensor(visible_df).float().numpy()

    y_true = label_np.flatten()
    y_score = pred_result.flatten()

    AUC = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    y_pred = (y_score >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp)

    logger.info('\n' + '='*60)
    logger.info(
        'OVERALL  AUC: {:.4f}  AUPR: {:.4f}  '
        'Acc: {:.4f}  F1: {:.4f}  '
        'Prec: {:.4f}  Recall: {:.4f}'.format(
            AUC, aupr, acc, f1, pre, rec))
    logger.info('='*60)

    pd.DataFrame(pred_result).to_csv(
        os.path.join(args.saved_path, 'result_gat.csv'),
        index=False, header=False)
    plot_result_auc(args,  label_np.flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, label_np.flatten(), pred_result.flatten(), aupr)

    # ── (B) Hidden-link recovery evaluation ────────────────────────────────
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

    # ── Save hidden-link report ────────────────────────────────────────────
    report_path = os.path.join(args.saved_path, 'hidden_link_report.txt')
    with open(report_path, 'w') as fh:
        fh.write('=== Hidden Link Recovery Report (GAT) ===\n')
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
    logger.info('\nDone.  Results saved to: {}'.format(args.saved_path))


if __name__ == '__main__':
    train()