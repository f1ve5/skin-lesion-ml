"""Portable commands that consume feature CSVs rather than image folders."""
import argparse
from pathlib import Path

import joblib

from .data import load_table, make_demo
from .evaluation import evaluate, save_protocol, tuned_fit
from .models import MODEL_NAMES, model_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate synthetic grouped feature data")
    demo.add_argument("--output", default="data/demo_features.csv")
    demo.add_argument("--seed", type=int, default=42)
    for command in ("evaluate", "train"):
        sub = commands.add_parser(command)
        sub.add_argument("--data", required=True)
        sub.add_argument("--output", default=f"outputs/{command}")
        sub.add_argument("--group-column", default="image_id")
        sub.add_argument("--label-column", default="dx")
        sub.add_argument("--inner-folds", type=int, default=4)
        sub.add_argument("--k", type=int, default=30)
        sub.add_argument("--selection", choices=["f", "mrmr"], default="mrmr")
        sub.add_argument("--seed", type=int, default=42)
        if command == "evaluate":
            sub.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                             default=["logreg", "svm", "random-forest", "gradient-boosting", "knn", "hierarchical"])
            sub.add_argument("--folds", type=int, default=5)
        else:
            sub.add_argument("--model", choices=MODEL_NAMES, default="hierarchical")
    predict = commands.add_parser("predict")
    predict.add_argument("--data", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", default="outputs/predictions.csv")
    args = parser.parse_args()
    if args.command == "demo":
        make_demo(args.output, args.seed)
        print(f"Synthetic demo data: {args.output}")
        return
    if args.command == "predict":
        bundle = joblib.load(args.model)
        frame, X = load_table(args.data, feature_names=bundle["feature_names"])
        result = frame[[col for col in ("image_id", "aug_tag", "filename") if col in frame]].copy()
        result["prediction"] = bundle["estimator"].predict(X)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(f"Predictions: {args.output}")
        return
    frame, X = load_table(args.data, args.group_column, args.label_column)
    y, groups = frame[args.label_column], frame[args.group_column]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        summary = evaluate(X, y, groups, args.models, output, args.folds, args.inner_folds,
                           args.k, args.selection, args.seed)
        print(summary.to_string(index=False))
    else:
        estimator, grid = model_spec(args.model, args.k, args.selection, args.seed)
        search = tuned_fit(estimator, grid, X, y, groups, args.inner_folds)
        joblib.dump({"estimator": search.best_estimator_, "feature_names": list(X.columns),
                     "best_params": search.best_params_}, output / "model.joblib")
        print(f"Final model (fit on all input rows): {output / 'model.joblib'}")
    save_protocol(output / "protocol.json", args.data, args, X, groups)
