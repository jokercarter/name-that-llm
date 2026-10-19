import numpy as np
import pandas as pd

from ntl_pact18.config import load_config
from ntl_pact18.data import make_synthetic_dataset
from ntl_pact18.labels import LABEL_TO_INDEX, N_CLASSES
from ntl_pact18.pipeline import PACT18Pipeline


def _tiny_config():
    return load_config(
        overrides=[
            "project.seed=3",
            "data.n_folds=2",
            "features.stylegram.char.max_features=3000",
            "features.stylegram.word.max_features=2500",
            "features.stylegram.shape.max_features=1500",
            "features.stylegram.function.max_features=1200",
            "features.stylegram.word.ngram=[1,2]",
            "features.stylegram.shape.ngram=[2,4]",
            "features.stylegram.function.ngram=[1,2]",
            "features.stylometry.n_estimators=40",
            "features.linear_baseline.C=1.0",
            "clustering.threshold=0.9",
            "evaluation.bootstrap=50",
        ]
    )


def test_pipeline_fits_and_predicts(tmp_path):
    paths = make_synthetic_dataset(
        tmp_path / "data", n_train_prompts=10, n_test_prompts=5, seed=5,
        test_missing_rate=0.1, min_words=40, max_words=90,
    )
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    truth = pd.read_csv(paths["test_truth"])

    cfg = _tiny_config()
    pipeline = PACT18Pipeline(cfg).fit(train, verbose=False)
    pred = pipeline.predict(test, verbose=False)

    assert pred.probs.shape == (len(test), N_CLASSES)
    assert pred.preds.shape == (len(test),)
    assert np.allclose(pred.probs.sum(axis=1), 1.0, atol=1e-6)
    assert set(pred.stage_preds) == {"blend", "em", "experts", "shield", "prism"}

    y = np.asarray([LABEL_TO_INDEX[v] for v in truth["model"]], dtype=int)
    acc = float((pred.preds == y).mean())
    # the synthetic task is solvable well above the 1/18 chance level
    assert acc > 0.15

    # weights are a probability distribution
    assert abs(sum(pipeline.weights.values()) - 1.0) < 1e-6

    # persistence round-trip
    path = pipeline.save(tmp_path / "model.joblib")
    reloaded = PACT18Pipeline.load(path)
    pred2 = reloaded.predict(test, verbose=False)
    assert np.array_equal(pred.preds, pred2.preds)


def test_blended_oof_is_available_after_fit(tmp_path):
    paths = make_synthetic_dataset(
        tmp_path / "d", n_train_prompts=8, n_test_prompts=3, seed=9,
        test_missing_rate=0.0, min_words=40, max_words=80,
    )
    train = pd.read_csv(paths["train"])
    pipeline = PACT18Pipeline(_tiny_config()).fit(train, verbose=False)
    oof = pipeline.blended_oof()
    assert oof.shape == (len(train), N_CLASSES)
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-6)
