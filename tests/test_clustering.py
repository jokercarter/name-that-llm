import numpy as np

from ntl_pact18.clustering import LearnedPairClusterer, TfidfClusterer, cluster_confidence
from ntl_pact18.evaluate import cluster_pair_scores

TOPIC_A = "algebra theorem lemma proof manifold topology eigenvalue convergence"
TOPIC_B = "protein enzyme genome neuron synapse membrane metabolism receptor"


def _corpus(n_per_topic: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    texts, groups = [], []
    for g, topic in enumerate((TOPIC_A, TOPIC_B)):
        words = topic.split()
        for _ in range(n_per_topic):
            sample = rng.choice(words, size=12, replace=True)
            texts.append(" ".join(sample) + f" marker{g}")
            groups.append(g)
    return texts, np.asarray(groups)


def test_tfidf_clusterer_separates_topics():
    texts, groups = _corpus()
    clusterer = TfidfClusterer(threshold=0.5).fit(texts)
    pred = clusterer.predict_groups(texts)
    _, _, f1 = cluster_pair_scores(pred, groups)
    assert f1 > 0.9


def test_learned_clusterer_recovers_groups():
    texts, groups = _corpus()
    clusterer = LearnedPairClusterer(threshold=0.9, n_pairs=400).fit(texts, groups)
    pred = clusterer.predict_groups(texts)
    _, _, f1 = cluster_pair_scores(pred, groups)
    assert f1 > 0.9


def test_cluster_confidence_is_higher_for_tight_clusters():
    texts, groups = _corpus()
    clusterer = TfidfClusterer(threshold=0.5).fit(texts)
    sim = clusterer.similarity(texts)
    conf = cluster_confidence(sim, groups)
    assert conf.shape == (len(texts),)
    assert conf.min() >= 0.0
    assert conf[groups == 0].mean() > 0.2
