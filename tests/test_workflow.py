import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from skin_lesion_ml.data import load_table, make_demo
from skin_lesion_ml.evaluation import grouped_splits, tuned_fit
from skin_lesion_ml.features import FeatureTransform
from skin_lesion_ml.models import HierarchicalClassifier, model_spec


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "features.csv"
        make_demo(self.path)
        self.frame, self.X = load_table(self.path)
        self.y = self.frame.dx.to_numpy()
        self.groups = self.frame.image_id.to_numpy()

    def test_group_isolation_and_coverage(self):
        counts = np.zeros(len(self.X), dtype=int)
        for train, test in grouped_splits(self.X, self.y, self.groups, 3):
            self.assertFalse(set(self.groups[train]) & set(self.groups[test]))
            counts[test] += 1
        np.testing.assert_array_equal(counts, 1)

    def test_metadata_excluded_and_conflicting_labels_rejected(self):
        self.assertEqual(self.X.shape[1], 18)
        self.frame.loc[0, "dx"] = "nv"
        self.frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "exactly one diagnosis"):
            load_table(self.path)

    def test_missing_and_infinite_columns(self):
        self.frame.loc[0, self.X.columns[0]] = np.inf
        self.frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, "Infinite"):
            load_table(self.path)
        with self.assertRaisesRegex(ValueError, "Missing model features"):
            load_table(self.path, feature_names=["unknown_feature"])

    def test_missing_constant_and_single_feature_handling(self):
        X = self.X.copy()
        X.loc[X.index[:5], X.columns[0]] = np.nan
        X["empty"] = np.nan
        X["constant"] = 1
        for selection in ("f", "mrmr"):
            transform = FeatureTransform(k=5, selection=selection, pca=True).fit(X, self.y)
            self.assertTrue(np.isfinite(transform.transform(X)).all())
        one = self.X.iloc[:, :1]
        transform = FeatureTransform(k=30).fit(one, self.y)
        self.assertEqual(transform.transform(one).shape, (len(one), 1))

    def test_hierarchy_predicts_without_labels_and_is_batch_invariant(self):
        model = HierarchicalClassifier(k=8, selection="f")
        clone(model)  # estimator must support inner-CV cloning
        model.fit(self.X, self.y)
        prediction = model.predict(self.X)
        self.assertTrue(set(prediction) <= {"mel", "nv", "vasc"})
        positions = np.arange(0, len(self.X), 7)
        np.testing.assert_array_equal(prediction[positions], model.predict(self.X.iloc[positions]))
        transform = model.layer2_.named_steps["features"]
        np.testing.assert_allclose(transform.transform(self.X)[positions], transform.transform(self.X.iloc[positions]))
        model_path = Path(self.temp.name) / "model.joblib"
        joblib.dump(model, model_path)
        restored = joblib.load(model_path)
        np.testing.assert_array_equal(prediction, restored.predict(self.X[self.X.columns[::-1]]))
        # A prediction CSV can omit every metadata and label column.
        self.X.to_csv(self.path, index=False)
        _, inference = load_table(self.path, feature_names=list(self.X.columns))
        np.testing.assert_array_equal(prediction, restored.predict(inference))

    def test_inner_search_refits_feature_processing(self):
        seen = []
        original = FeatureTransform.fit
        def observed(transform, X, y):
            seen.append(set(X.index))
            return original(transform, X, y)
        train, test = grouped_splits(self.X, self.y, self.groups, 3)[0]
        model, grid = model_spec("logreg", k=8, selection="f")
        with patch.object(FeatureTransform, "fit", observed):
            search = tuned_fit(model, grid, self.X.iloc[train], self.y[train], self.groups[train], 2)
        self.assertEqual(len(seen), 5)  # 2 candidates x 2 inner folds + refit
        self.assertTrue(all(ids <= set(train) for ids in seen))
        self.assertTrue(all(len(ids) < len(train) for ids in seen[:-1]))
        self.assertFalse(any(ids & set(test) for ids in seen))
        self.assertEqual(len(search.predict(self.X.iloc[test])), len(test))

    def test_reject_unusable_fold_counts(self):
        with self.assertRaises(ValueError):
            grouped_splits(self.X, self.y, self.groups, 1)
        with self.assertRaises(ValueError):
            grouped_splits(self.X, self.y, self.groups, 100)


if __name__ == "__main__":
    unittest.main()
