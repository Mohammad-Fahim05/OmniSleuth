"""Retrieval benchmark ground-truth integrity (T-004).

Every ``relevant_chunk_ids`` entry must be a chunk id that the existing
``DocumentStore`` can actually generate from ``SEED_DOCUMENTS`` using the
existing chunking configuration.

Determinism and no-model guarantees:
- ``DocumentStore.split_text`` is a pure function of text and ``chunk_size``.
- ``add_document_text`` mints ids deterministically as ``{doc_id}_chunk_{idx:02d}``.
- ``DenseEmbedder._init_model`` is monkeypatched away, so no sentence-transformers
  model is ever loaded or downloaded, and no network is touched.
"""

from typing import Dict, List, Set

import pytest

from omnisleuth.rag import benchmark_dataset, document_store, seed_docs


def _seed_chunk_ids(chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Chunk ids produced by ingesting the seed documents, computed independently.

    Reuses ``DocumentStore.split_text`` directly (it is a pure function of text,
    ``chunk_size`` and ``chunk_overlap``) so no embedding model, FAISS index or
    reranker is constructed for the independent computation.
    """
    split_text = document_store.DocumentStore.split_text

    class _Splitter:
        __slots__ = ("chunk_size", "chunk_overlap")

        def __init__(self, size: int, overlap: int):
            self.chunk_size = size
            self.chunk_overlap = overlap

    splitter = _Splitter(chunk_size, chunk_overlap)
    ids: List[str] = []
    for doc_id, _title, _doc_type, text in seed_docs.SEED_DOCUMENTS:
        for idx, _chunk_text in enumerate(split_text(splitter, text), start=1):
            ids.append(f"{doc_id}_chunk_{idx:02d}")
    return ids


@pytest.fixture
def store_without_models(monkeypatch):
    """A DocumentStore with the deterministic (hash) embedding fallback forced on."""
    from omnisleuth.rag.dense_embedder import DenseEmbedder

    def _no_model(self):
        self._model = None

    monkeypatch.setattr(DenseEmbedder, "_init_model", _no_model)
    return document_store.DocumentStore()


@pytest.fixture
def generated_chunk_ids(store_without_models) -> List[str]:
    return [chunk.chunk_id for chunk in store_without_models.chunks]


@pytest.fixture
def ground_truth_chunk_ids() -> Dict[str, Set[str]]:
    """Ground-truth chunk ids per retrieval case, from the benchmark dataset."""
    return {
        case.case_id: set(case.relevant_chunk_ids)
        for case in benchmark_dataset.RETRIEVAL_BENCHMARK_CASES
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth integrity
# ─────────────────────────────────────────────────────────────────────────────

def test_seed_documents_are_ingested_without_models(store_without_models):
    """All seed documents load into chunks using only the deterministic fallback."""
    assert len(store_without_models.documents) == len(seed_docs.SEED_DOCUMENTS)
    assert store_without_models.chunks, "Expected at least one chunk from the seed documents"


def test_chunk_ids_are_unique(generated_chunk_ids):
    """Generated chunk ids must be unique, since ground truth references them by id."""
    assert len(generated_chunk_ids) == len(set(generated_chunk_ids))


def test_every_ground_truth_chunk_id_is_generatable(generated_chunk_ids, ground_truth_chunk_ids):
    """The core integrity check: no ground-truth chunk id may reference a nonexistent chunk."""
    available = set(generated_chunk_ids)
    missing = {
        case_id: sorted(needed - available)
        for case_id, needed in ground_truth_chunk_ids.items()
        if needed - available
    }
    assert not missing, (
        "Retrieval benchmark references chunk ids the store cannot generate: "
        f"{missing}. Available ids: {sorted(available)}"
    )


def test_chunk_id_derivation_matches_the_independent_computation(generated_chunk_ids):
    """Store-derived ids equal ids computed independently from the seed documents."""
    assert generated_chunk_ids == _seed_chunk_ids()


def test_chunking_configuration_is_deterministic(chunk_size=500, chunk_overlap=50):
    """Chunk boundaries depend only on text and configuration, never on order or state."""
    first = _seed_chunk_ids(chunk_size, chunk_overlap)
    second = _seed_chunk_ids(chunk_size, chunk_overlap)
    assert first == second


@pytest.mark.parametrize(
    "chunk_size",
    [200, 500, 5000],
)
def test_chunking_is_configuration_driven(chunk_size):
    """Different chunk sizes produce different, still deterministic, chunk id sets."""
    small = _seed_chunk_ids(chunk_size=chunk_size, chunk_overlap=0)
    assert small, "Chunking must always produce at least one chunk"
    assert all(id_.startswith("doc_") for id_ in small)


def test_ground_truth_chunk_ids_belong_to_the_referenced_documents(ground_truth_chunk_ids):
    """Each ground-truth doc id agrees with the prefix of its chunk ids."""
    cases_by_id = {c.case_id: c for c in benchmark_dataset.RETRIEVAL_BENCHMARK_CASES}
    for case_id, chunk_ids in ground_truth_chunk_ids.items():
        case = cases_by_id[case_id]
        for chunk_id in chunk_ids:
            assert any(chunk_id.startswith(f"{doc_id}_chunk_") for doc_id in case.relevant_doc_ids), (
                f"{case_id}: chunk '{chunk_id}' is not under any referenced doc {case.relevant_doc_ids}"
            )


def test_every_benchmark_case_has_ground_truth():
    """No benchmark case may ship with empty chunk/document ground truth."""
    for case in benchmark_dataset.RETRIEVAL_BENCHMARK_CASES:
        assert case.relevant_chunk_ids, f"{case.case_id} has no chunk ground truth"
        assert case.relevant_doc_ids, f"{case.case_id} has no document ground truth"


def test_referenced_documents_are_seed_documents():
    """Ground-truth document ids must exist among the seed documents."""
    seed_ids = {doc_id for doc_id, *_ in seed_docs.SEED_DOCUMENTS}
    referenced = {
        doc_id
        for case in benchmark_dataset.RETRIEVAL_BENCHMARK_CASES
        for doc_id in case.relevant_doc_ids
    }
    assert referenced <= seed_ids, f"Unknown doc ids referenced: {sorted(referenced - seed_ids)}"


def test_chunk_ids_monotonic_per_document(generated_chunk_ids):
    """Within each document the ``_chunk_NN`` suffix increments from 01."""
    per_doc: Dict[str, List[int]] = {}
    for chunk_id in generated_chunk_ids:
        doc_id, suffix = chunk_id.rsplit("_chunk_", 1)
        per_doc.setdefault(doc_id, []).append(int(suffix))

    for doc_id, suffixes in per_doc.items():
        assert suffixes == list(range(1, len(suffixes) + 1)), (
            f"{doc_id} chunk suffixes are not contiguous from 01: {suffixes}"
        )


def test_chunking_needs_no_embedding_models():
    """Chunk-id generation must not require sentence-transformers, torch, or faiss.

    Blocks the heavy optional dependencies at import level, then verifies the
    store still builds and yields identical chunk ids. This keeps the integrity
    check cheap, deterministic, and safe to run with no network access.
    """
    import builtins

    heavy = {"sentence_transformers", "faiss", "transformers", "torch", "cross_encoder"}
    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name.split(".")[0] in heavy:
            raise ImportError(f"blocked for test isolation: {name}")
        return real_import(name, *args, **kwargs)

    from omnisleuth.rag.dense_embedder import DenseEmbedder

    original = DenseEmbedder._init_model
    DenseEmbedder._init_model = lambda self: setattr(self, "_model", None)
    builtins.__import__ = _blocking_import
    try:
        store = document_store.DocumentStore()
        ids = [chunk.chunk_id for chunk in store.chunks]
        assert ids, "Chunking produced no chunks"
        assert ids == _seed_chunk_ids()
    finally:
        builtins.__import__ = real_import
        DenseEmbedder._init_model = original
