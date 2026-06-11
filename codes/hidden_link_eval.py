"""
hidden_link_eval.py
===================

Hidden Link Recovery Evaluation — Extension layer for the existing framework.

Design principle (Open/Closed):
  - Does NOT modify Model, ModelGAT, train loops, or evaluation functions.
  - Wraps the existing pipeline: hide links → train → evaluate recovery.

Evaluation philosophy
---------------------
The experiment asks a single question:
    "After training WITHOUT a set of known positive links,
     does the model assign high scores to those hidden links?"

This is a RECOVERY task, not a binary classification task.
We therefore report only direct recovery metrics:

    Recovery Rate @ threshold  — fraction of hidden links the model
                                 scored at or above `threshold`
    Mean Hidden Score          — average sigmoid score over all hidden links
    Median Hidden Score        — median sigmoid score over all hidden links

No negative sampling. No AUC / AP / F1 / Precision / Recall.
Those remain in the standard cross-validation pipeline, untouched.

Controllable parameters (pass via args or function kwargs)
----------------------------------------------------------
    hidden_ratio   float in (0, 1)   fraction of positives to hide
    threshold      float in (0, 1)   score cutoff for "recovered"

Usage
-----
    hidden_links, visible_df = hide_positive_links(df, hidden_ratio, seed)
    report = evaluate_hidden_link_recovery(model, g, g_llm, feature,
                                           hidden_links, device, logger,
                                           threshold)
"""

import numpy as np
import torch as th
import logging


# ---------------------------------------------------------------------------
# Step 1 — Hide a fraction of positive links before training
# ---------------------------------------------------------------------------

def hide_positive_links(
    df: np.ndarray,
    hidden_ratio: float = 0.10,
    seed: int = 0,
) -> tuple:
    """
    Randomly select a fraction of positive (value==1) entries in the DDA
    matrix and remove them, returning both the hidden coordinates and the
    modified matrix.

    The hidden links are stored separately and NEVER reintroduced into the
    graph or label matrix used during training / k-fold evaluation.

    Parameters
    ----------
    df            : full DDA matrix  (shape: n_drugs × n_diseases)
    hidden_ratio  : fraction of positives to hide  — controllable via
                    --hidden_ratio CLI arg  (default 0.10 = 10 %)
    seed          : random seed for reproducibility

    Returns
    -------
    hidden_links  : np.ndarray  shape (k, 2)  — row/col indices of hidden links
    visible_df    : np.ndarray  same shape as df, hidden entries set to 0
    """
    if not (0.0 < hidden_ratio < 1.0):
        raise ValueError(
            "hidden_ratio must be in (0, 1). Got: {}".format(hidden_ratio)
        )

    rng = np.random.default_rng(seed)

    pos_coords = np.array(np.where(df == 1)).T     # shape: (n_pos, 2)
    n_pos      = len(pos_coords)
    n_hide     = max(1, int(round(n_pos * hidden_ratio)))

    chosen_idx   = rng.choice(n_pos, size=n_hide, replace=False)
    hidden_links = pos_coords[chosen_idx]           # shape: (n_hide, 2)

    visible_df = df.copy()
    visible_df[hidden_links[:, 0], hidden_links[:, 1]] = 0

    return hidden_links, visible_df


# ---------------------------------------------------------------------------
# Step 2 — Evaluate hidden-link recovery after training is complete
# ---------------------------------------------------------------------------

@th.no_grad()
def evaluate_hidden_link_recovery(
    model,
    g,
    g_llm,
    feature: dict,
    hidden_links: np.ndarray,
    device: th.device,
    logger: logging.Logger = None,
    threshold: float = 0.5,
) -> dict:
    """
    Score the hidden positive links using the ALREADY TRAINED model and
    report how well it recovers them.

    No negative sampling is performed.  Recovery is measured purely on the
    hidden positive coordinates:

        Recovery Rate @ threshold  = #{score >= threshold} / n_hidden
        Mean Hidden Score          = mean of sigmoid scores over hidden links
        Median Hidden Score        = median of sigmoid scores over hidden links

    Parameters
    ----------
    model        : trained Model or ModelGAT instance (already .eval())
    g            : DGL training graph (hidden links absent)
    g_llm        : DGL LLM graph (hidden links absent)
    feature      : feature dict used during training
    hidden_links : np.ndarray shape (k, 2) from hide_positive_links()
    device       : torch device
    logger       : optional logger; prints to stdout if None
    threshold    : score cutoff for counting a link as "recovered"
                   — controllable via --threshold CLI arg  (default 0.5)

    Returns
    -------
    report : dict with keys
        n_hidden, hidden_ratio_requested,
        recovery_rate, mean_score_hidden, median_score_hidden,
        n_recovered, threshold,
        scores_hidden  (raw sigmoid scores — shape: (k,))
    """

    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    model.eval()

    # ------------------------------------------------------------------ #
    # Full score matrix (sigmoid over logits)                             #
    # ------------------------------------------------------------------ #
    score_matrix = model([g, g_llm], feature)                    # logits
    pred_matrix  = th.sigmoid(score_matrix).cpu().numpy()        # (n_drugs, n_diseases)

    # ------------------------------------------------------------------ #
    # Extract scores ONLY for the hidden positive links                   #
    # ------------------------------------------------------------------ #
    hd_rows       = hidden_links[:, 0]
    hd_cols       = hidden_links[:, 1]
    scores_hidden = pred_matrix[hd_rows, hd_cols]                # shape: (k,)

    n_hidden    = len(scores_hidden)
    n_recovered = int((scores_hidden >= threshold).sum())

    recovery_rate      = n_recovered / n_hidden
    mean_hidden_score  = float(scores_hidden.mean())
    median_hidden_score = float(np.median(scores_hidden))

    report = dict(
        n_hidden             = n_hidden,
        n_recovered          = n_recovered,
        threshold            = threshold,
        recovery_rate        = recovery_rate,
        mean_score_hidden    = mean_hidden_score,
        median_score_hidden  = median_hidden_score,
        scores_hidden        = scores_hidden,
    )

    # ------------------------------------------------------------------ #
    # Print report                                                         #
    # ------------------------------------------------------------------ #
    _log('\n' + '=' * 60)
    _log('HIDDEN LINK RECOVERY EVALUATION')
    _log('=' * 60)
    _log('  Hidden links (positives removed before training) : {:d}'.format(n_hidden))
    _log('  Threshold used for recovery decision             : {:.2f}'.format(threshold))
    _log('  Recovered (score >= threshold)                   : {:d} / {:d}'.format(
        n_recovered, n_hidden))
    _log('  Recovery Rate @ {:.2f}                          : {:.4f}  ({:.1f}%)'.format(
        threshold, recovery_rate, recovery_rate * 100))
    _log('  Mean   score on hidden positives                 : {:.4f}'.format(mean_hidden_score))
    _log('  Median score on hidden positives                 : {:.4f}'.format(median_hidden_score))
    _log('=' * 60 + '\n')

    return report