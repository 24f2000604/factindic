"""Stage 4 single-round candidate reranking and capped pruning (spec §8.4.3–4), plus the
drop-only repair baseline (Exp 5a)."""
from __future__ import annotations

from .normalizer import anchor_recall_salient, anchors, salient_anchors
from .text import W


def J_score(cand_summary, verdicts, rec, tau, lam, ner=None):
    """J(S_k) = (frac claims with p ≥ τ) · R_anchor(S_k, D) − λ_len·|W(S_k) − L|/L."""
    frac = (sum(v["p"] >= tau for v in verdicts) / len(verdicts)) if verdicts else 0.0
    a_s = anchors(cand_summary, rec["lang"], ner.entities(cand_summary) if ner else None)
    sal = salient_anchors(rec["anchors"], rec["text"], rec["lang"])
    r_anchor = anchor_recall_salient(a_s, rec["anchors"], sal, rec["text"], rec["lang"])
    L = rec["L_target"]
    j = frac * r_anchor - lam * abs(W(cand_summary) - L) / L
    return dict(J=j, frac_supported=frac, R_anchor=r_anchor, words=W(cand_summary))


def prune(verdicts, tau):
    """Drop unsupported sentences only if at least one verified sentence remains (single pass)."""
    keep = [v["claim"] for v in verdicts if v["p"] >= tau]
    if keep and len(keep) < len(verdicts):
        return " ".join(keep), True
    return " ".join(v["claim"] for v in verdicts), False


def drop_only(verdicts, tau):
    """Delete every sentence the verifier rejects, even if nothing is left (recall-collapse baseline)."""
    return " ".join(v["claim"] for v in verdicts if v["p"] >= tau)
