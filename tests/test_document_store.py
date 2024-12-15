import pytest
from django.db import connection
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from pgvector.django import L2Distance
from testapp.models import BasicDocument, FullDocument

from django_haystack.document_store import DjangoModelDocumentStore


@pytest.fixture
def document_store():
    return DjangoModelDocumentStore(model=BasicDocument)


@pytest.fixture
def full_document_store():
    return DjangoModelDocumentStore(model=FullDocument)


@pytest.fixture
def sample_documents():
    return [
        Document(
            id="doc1",
            content="test content 1",
            meta={"key1": "value1"},
        ),
        Document(
            id="doc2",
            content="test content 2",
            meta={"key2": "value2"},
        ),
    ]


@pytest.fixture
def sample_documents_with_embeddings():
    return [
        Document(
            id="doc1",
            content="test content 1",
            meta={"key1": "value1"},
            embedding=[1.0, 0.0, 0.0],
        ),
        Document(
            id="doc2",
            content="test content 2",
            meta={"key2": "value2"},
            embedding=[0.0, 1.0, 0.0],
        ),
    ]


@pytest.mark.django_db
class TestDjangoModelDocumentStore:
    def test_init(self, document_store):
        assert document_store.model == BasicDocument
        assert document_store.language == "english"

    def test_count_documents_empty(self, document_store):
        assert document_store.count_documents() == 0

    def test_count_documents(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)
        assert document_store.count_documents() == 2

    def test_write_documents(self, document_store, sample_documents):
        num_written = document_store.write_documents(sample_documents)
        assert num_written == 2
        
        docs = document_store.filter_documents()
        assert len(docs) == 2
        assert {d.id for d in docs} == {"doc1", "doc2"}

    def test_write_documents_duplicate_fail(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)
        
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(
                sample_documents, 
                policy=DuplicatePolicy.FAIL
            )

    def test_write_documents_duplicate_skip(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)
        
        num_written = document_store.write_documents(
            sample_documents,
            policy=DuplicatePolicy.SKIP
        )
        assert num_written == 0
        assert document_store.count_documents() == 2

    def test_write_documents_duplicate_overwrite(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)
        
        modified_docs = [
            Document(
                id="doc1",
                content="modified content",
                meta={"key": "modified"}
            )
        ]
        
        num_written = document_store.write_documents(
            modified_docs,
            policy=DuplicatePolicy.OVERWRITE
        )
        assert num_written == 1
        
        docs = document_store.filter_documents()
        modified_doc = next(d for d in docs if d.id == "doc1")
        assert modified_doc.content == "modified content"
        assert modified_doc.meta == {"key": "modified"}

    def test_delete_documents(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)
        assert document_store.count_documents() == 2
        
        document_store.delete_documents(["doc1"])
        assert document_store.count_documents() == 1
        
        docs = document_store.filter_documents()
        assert docs[0].id == "doc2"

    def test_filter_documents(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)
        
        filters = {
            "operator": "AND",
            "conditions": [
                {
                    "field": "meta.key1",
                    "operator": "==",
                    "value": "value1"
                }
            ]
        }
        docs = document_store.filter_documents(filters)
        assert len(docs) == 1
        assert docs[0].id == "doc1"

    @pytest.mark.django_db(transaction=True)
    def test_embedding_retrieval(self, full_document_store, sample_documents_with_embeddings):
        full_document_store.write_documents(sample_documents_with_embeddings)
        
        # Query vector more similar to doc1
        query_embedding = [0.9, 0.1, 0.0]
        
        results = full_document_store.embedding_retrieval(
            query_embedding=query_embedding,
            top_k=1,
            vector_function=L2Distance
        )
        
        assert len(results) == 1
        assert results[0].id == "doc1"
        assert hasattr(results[0], "score")

    def test_keyword_retrieval(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)
        
        results = document_store.keyword_retrieval(
            query="content 1",
            top_k=1
        )
        
        assert len(results) == 1
        assert results[0].id == "doc1"
        assert hasattr(results[0], "score")
