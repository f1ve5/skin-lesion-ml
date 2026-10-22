"""Grouped nested CV, auditable out-of-fold predictions, and local artifacts."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from .data import CLASSES
from .models import model_spec


def grouped_splits(X, y, groups, folds):
    if folds < 2 or len(np.unique(groups)) < folds:
        raise ValueError(f"Need at least {folds} independent groups and folds >= 2.")
    splits = list(GroupKFold(folds).split(X, y, groups))
    for train, test in splits:
        if set(np.asarray(y)[train]) != set(CLASSES):
            raise ValueError("A training fold lacks a class; use fewer folds or more groups per class.")
        if set(np.asarray(groups)[train]) & set(np.asarray(groups)[test]):
            raise AssertionError("Group leakage detected.")
    return splits


def metrics(y, prediction):
    return {"accuracy": accuracy_score(y, prediction),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "f1_macro": f1_score(y, prediction, labels=CLASSES, average="macro", zero_division=0)}


def tuned_fit(model, grid, X, y, groups, folds):
    splits = grouped_splits(X, y, groups, folds)
    # Splits contain only the current outer training rows. The whole estimator,
    # including screening/PCA/interactions, is refitted per inner training fold.
    search = GridSearchCV(model, grid, cv=splits, scoring="f1_macro", n_jobs=1, error_score="raise")
    search.fit(X, y)
    return search


def evaluate(X, y, groups, names, output, folds=5, inner_folds=4, k=30, selection="mrmr", seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    y, groups = np.asarray(y), np.asarray(groups)
    outer = grouped_splits(X, y, groups, folds)
    rows, fold_rows, group_rows, assignments = [], [], [], []
    for fold, (_, test) in enumerate(outer, 1):
        assignments.extend({"row_index": int(i), "group": groups[i], "fold": fold} for i in test)
    pd.DataFrame(assignments).sort_values("row_index").to_csv(output / "fold_assignments.csv", index=False)
    # Validate every inner split before launching expensive model searches.
    for train, _ in outer:
        grouped_splits(X.iloc[train], y[train], groups[train], inner_folds)
    for name in names:
        prediction = np.empty(len(y), dtype=object)
        for fold, (train, test) in enumerate(outer, 1):
            print(f"{name}: outer fold {fold}/{folds}", flush=True)
            model, grid = model_spec(name, k, selection, seed)
            search = tuned_fit(model, grid, X.iloc[train], y[train], groups[train], inner_folds)
            prediction[test] = search.predict(X.iloc[test])
            fold_rows.append({"model": name, "fold": fold, **metrics(y[test], prediction[test]),
                              "inner_best_f1": search.best_score_, "best_params": json.dumps(search.best_params_)})
        rows.append({"model": name, **metrics(y, prediction)})
        oof = pd.DataFrame({"row_index": np.arange(len(y)), "group": groups, "y_true": y, "y_pred": prediction})
        oof.to_csv(output / f"{name}_oof.csv", index=False)
        # Majority vote gives one evaluation unit per independent group.
        # Ties resolve by sorted class order; record this policy in the protocol.
        group_true = oof.groupby("group").y_true.first()
        group_pred = oof.groupby("group").y_pred.agg(lambda s: s.mode().iloc[0])
        group_rows.append({"model": name, **metrics(group_true, group_pred)})
        report = classification_report(y, prediction, labels=CLASSES, output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(output / f"{name}_classification_report.csv")
        cm = pd.DataFrame(confusion_matrix(y, prediction, labels=CLASSES), index=CLASSES, columns=CLASSES)
        cm.to_csv(output / f"{name}_confusion_matrix.csv")
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output / "group_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ax = summary.set_index("model").plot.bar(figsize=(11, 5), ylim=(0, 1), rot=25)
    ax.set_ylabel("OOF score (sample level)")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "comparison.png", dpi=160)
    plt.close(ax.figure)
    return summary


def save_protocol(path, csv_path, args, X, groups):
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "matplotlib", "joblib")}
    protocol = {"arguments": vars(args), "versions": versions, "rows": len(X),
                "python": sys.version,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path(__file__).parent.glob("*.py"))},
                "features": X.shape[1], "groups": len(np.unique(groups)),
                "data_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                "evaluation": "nested GroupKFold; pooled sample OOF and group majority-vote scores",
                "group_vote_tie_break": "lexicographically first class",
                "upstream_features": "Precomputed input; upstream fitting/leakage must be audited separately."}
    Path(path).write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
