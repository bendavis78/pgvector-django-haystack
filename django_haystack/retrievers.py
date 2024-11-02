from typing import Any, List

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy
from pgvector.django import functions
from pgvector.django.functions import BitDistanceBase, DistanceBase

from django_haystack.document_store import DjangoModelDocumentStore


@component
class DjangoModelEmbeddingRetriever:
    """
    Retrieves documents from the `DjangoModelDocumentStore`, based on their dense embeddings.

    Documents are ranked depending on the vector function used. The ranking score is stored in the
    `score` field of the resulting Document ojbects.
    """

    def __init__(
        self,
        document_store: DjangoModelDocumentStore,
        filters: dict[str, Any] = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
        vector_function: type[DistanceBase | BitDistanceBase] = None,
    ):
        """
        :param document_store: An instance of `PgvectorDocumentStore`.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param vector_function: The similarity function to use when searching for similar embeddings.
            Defaults to the one set in the `document_store` instance.
            **Important**: if the underlying document model is using the `"hnsw"` search index, the
            vector function should match the opclass utilized in the index.
        :param filter_policy: Policy to determine how filters are applied.
        """
        self.document_store = document_store
        self.vector_function = vector_function
        self.filters = filters or {}
        self.top_k = top_k

        if isinstance(filter_policy, FilterPolicy):
            self.filter_policy = filter_policy
        else:
            self.filter_policy = FilterPolicy.from_str(filter_policy)

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            vector_function=self.vector_function.__name__,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DjangoModelEmbeddingRetriever":
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = DjangoModelDocumentStore.from_dict(
            doc_store_params
        )

        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)

        if vector_function := data["init_parameters"].get("vector_function"):
            data["vector_function"] = getattr(functions, vector_function)

        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] = None,
        top_k: int = None,
        vector_function: type[DistanceBase | BitDistanceBase] = None,
    ):
        """
        Retrive documents from the `DjangoModelDocumentStore`, based on their dense embeddings.

        :param query_embedding: Query embedding to use for the document search.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are
                        applied depends on the `filter_policy` chosen at retriever initialization.
                        See init method docstring for more details.
        :param top_k: Maximum number of Documents to return.
        :param queryset: Optional queryset from which to retrieve documents. By default, the
                         document_store.get_queryset() method is used.
        :param vector_function: one of the vector functions from `pgvector.django.functions`

        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s that match the query.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        top_k = top_k or self.top_k
        vector_function = vector_function or self.vector_function

        queryset = self.document_store.embedding_retrieval(
            query_embedding,
            filters=filters,
            top_k=top_k,
            vector_function=vector_function,
        )

        return {"documents": [obj.to_haystack_document(score=obj.score) for obj in queryset]}


@component
class DjangoModelKeywordRetriever:
    """
    Retrieve documents from the `DjangoModelDocumentStore`, based on keywords.

    To rank the documents, the `django.contrib.postgres.search.SearchRank`
    function is used. It considers how often the query terms appear in the
    document, how close together the terms are in the document, and how
    important is the part of the document where they occur.

    For more details, see the
    [Django Documentation](https://docs.djangoproject.com/en/5.1/ref/contrib/postgres/search/)
    """  # noqa: E501

    def __init__(
        self,
        *,
        document_store: DjangoModelDocumentStore,
        filters: dict[str, Any] = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ):
        """
        :param document_store: An instance of `DjangoModelDocumentStore`.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.
        """
        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.filter_policy = (
            filter_policy
            if isinstance(filter_policy, FilterPolicy)
            else FilterPolicy.from_str(filter_policy)
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self.filters,
            top_k=self.top_k,
            filter_policy=self.filter_policy.value,
            document_store=self.document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DjangoModelKeywordRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        doc_store_params = data["init_parameters"]["document_store"]
        data["init_parameters"]["document_store"] = DjangoModelDocumentStore.from_dict(
            doc_store_params
        )
        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] = None,
        top_k: int = None,
    ):
        """
        Retrieve documents from the `DjangoModelDocumentStore`, based on keywords.

        :param query: String to search in `Document`s' content.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are
                        applied depends on the `filter_policy` chosen at retriever initialization.
                        See init method docstring for more details.
        :param top_k: Maximum number of Documents to return.
        :param queryset: Optional queryset from which to retrieve documents. By default, the
                         document_store.get_queryset() method is used.

        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s that match the query.
        """
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)

        queryset = self.document_store.keyword_retrieval(
            query,
            filters=filters,
            top_k=top_k or self.top_k,
        )

        return {"documents": [obj.to_haystack_document(score=obj.score) for obj in queryset]}
