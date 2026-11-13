from backend.app.core.cache import cache
from backend.app.nlp.embeddings import semantic_engine
from backend.app.nlp.tokenizer import clean_and_tokenize


def test_performance_cache():
    cache.clear()
    assert cache.get("sample text", "test") is None
    cache.set("sample text", 42, "test")
    assert cache.get("sample text", "test") == 42


def test_cached_tokenizer():
    cache.clear()
    text = "Building scalable microservices with Python and FastAPI."
    tokens_1 = clean_and_tokenize(text)
    # Check cache hit
    tokens_2 = clean_and_tokenize(text)
    assert tokens_1 == tokens_2
    assert "python" in tokens_2


def test_cached_semantic_similarity():
    cache.clear()
    jd = "Python engineer with PostgreSQL experience."
    res = "Senior Python developer specialized in PostgreSQL databases."
    sim_1 = semantic_engine.compute_similarity(jd, res)
    sim_2 = semantic_engine.compute_similarity(jd, res)
    assert sim_1 == sim_2
    assert sim_1 > 0.05
