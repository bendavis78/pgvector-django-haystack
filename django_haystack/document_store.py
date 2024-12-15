import logging
from typing import Any, Dict, List

from django.apps import apps
from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db import models, transaction
from django.db.models import ExpressionWrapper, F, Value
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from pgvector.django.functions import (
    BitDistanceBase,
    CosineDistance,
    DistanceBase,
    L1Distance,
    L2Distance,
    MaxInnerProduct,
)

from django_haystack.models import HaystackDocumentStoreModel, HaystackDocumentStoreQuerySet

log = logging.getLogger(__name__)


class DjangoModelDocumentStore:
    """
    Haystack Document store using a Django model as the backend to store Haystack documents.
    """

    def __init__(
        self,
        model: type[HaystackDocumentStoreModel],
        language: str = "english",
    ):
        """
        Creates a new DjangoModelDocumentStore instance.

        :param: model: The Django model to use as the document store. This model must be a subclass
                       of `HaystackDocumentStoreModel`.
        :param language: The language used for keyword-based retrieval.
        """
        self.model = model
        self.language = language or model._haystack.language

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": (self.model._meta.app_label, self.model._meta.model_name),
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DjangoModelDocumentStore":
        model = apps.get_model(*config["model"])
        return cls(model=model)

    def count_documents(self) -> int:
        """
        Returns the number of documents stored in the model.
        """
        return self.model.objects.count()

    def get_queryset(self) -> HaystackDocumentStoreQuerySet:
        """
        Returns the queryset for the model.
        """
        return self.model.objects.all()

    def filter_documents(self, filters: dict[str, Any] = None) -> list[Document]:
        """
        Returns the documents that match the filters provided.

        For a detailed specification of the filters,
        refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering)

        :param filters: The filters to apply to the document list.
        :raises ValueError: If `filters` is not a dictionary.
        :returns: A list of Documents that match the given filters.
        """
        queryset = self.get_queryset()

        if filters:
            queryset = queryset.apply_haystack_filters(filters)

        return [obj.to_haystack_document() for obj in queryset]

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ) -> int:
        """
        Writes or overwrites documents into the model.

        :param documents: A list of documents to write to the model.
        :param policy: The duplicate policy to use when writing documents.
        :return: The number of documents written.
        """
        num_written = 0

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL

        with transaction.atomic():
            for document in documents:
                lookup = {self.model._haystack.field_map["id"]: document.id}
                try:
                    existing = self.model.objects.get(**lookup)
                except self.model.DoesNotExist:
                    existing = None

                if existing and policy == DuplicatePolicy.OVERWRITE:
                    existing.delete()
                elif existing and policy == DuplicatePolicy.FAIL:
                    raise DuplicateDocumentError(f"Duplicate document found for id {document.id}")
                elif existing and policy == DuplicatePolicy.SKIP:
                    continue

                db_doc = self.model.from_haystack_document(document)
                db_doc.save()
                num_written += 1

        return num_written

    def delete_documents(self, document_ids: list[str] = None) -> None:
        """
        Deletes documents that match the provided document_ids.

        :param document_ids: The document ids to delete.
        """
        self.model.objects.filter(id__in=document_ids).delete()

    def embedding_retrieval(
        self,
        query_embedding: List[float],
        *,
        filters: Dict[str, Any] = None,
        top_k: int = None,
        vector_function: DistanceBase | BitDistanceBase = None,
        embedding_field: str = "embedding",
        queryset=None,
    ) -> HaystackDocumentStoreQuerySet:
        """
        Retrieves documents from the `DjangoModelDocumentStore`, based on their dense embeddings.

        Documents are ranked depending on the vector function used. The ranking score is stored in
        the `score` field of the resulting Document ojbects.

        :param query_embedding: Query embedding to use for the document search. :param filters:
                                Filters applied to the retrieved Documents. The way runtime
                                filters are applied depends on the `filter_policy` chosen at
                                retriever initialization. See init method docstring for more
                                details.
        :param top_k: Maximum number of Documents to return. :param queryset: Optional
                      queryset from which to retrieve documents. By default, the
                      document_store.get_queryset() method is used.
        :param vector_function: one of the vector functions from `pgvector.django.functions`

        :returns: HaystackDocumentStoreQuerySet with a `score` annotation ranked by similarity to
        `query_embedding`

        """
        vector_function = vector_function or self.model._haystack.vector_function
        if vector_function is None:
            raise ValueError(
                "A vector_function must be provided or defined on the model's HaystackOptions"
            )

        if queryset is None:
            queryset = self.get_queryset()

        score_expression = vector_function(F(embedding_field), query_embedding)
        if vector_function is CosineDistance:
            score_expression = Value("1") - score_expression
        elif vector_function is MaxInnerProduct:
            score_expression = score_expression * Value("-1")

        score_expression = ExpressionWrapper(score_expression, output_field=models.FloatField())

        queryset = queryset.annotate(score=score_expression)

        if vector_function in {L1Distance, L2Distance}:
            queryset = queryset.order_by("score")
        queryset = queryset.order_by("-score")

        if filters:
            queryset = queryset.apply_haystack_filters(filters)

        if top_k:
            queryset = queryset[:top_k]

        return queryset

    def keyword_retrieval(
        self,
        query: str,
        *,
        filters: Dict[str, Any] = None,
        top_k: int = None,
        queryset=None,
    ):
        """
        Retrieve documents from the `DjangoModelDocumentStore`, based on keywords.

        :param filters: Filters applied to the retrieved Documents. The way runtime filters are
                        applied depends on the `filter_policy` chosen at retriever initialization.
                        See init method docstring for more details.
        :param top_k: Maximum number of Documents to return.
        :param queryset: Optional queryset from which to retrieve documents. By default, the
                         document_store.get_queryset() method is used.

        :returns: HaystackDocumentStoreQuerySet with a `score` annotation ranked by similarity to
                  `query`
        """
        if queryset is None:
            queryset = self.get_queryset()

        content_field = queryset.model._haystack.get_field("content")
        top_k = top_k or self.top_k

        search_query = SearchQuery(query, config=self.language)
        search_vector = SearchVector(content_field.name, config=self.language)

        queryset = queryset.annotate(score=SearchRank(search_vector, search_query))
        queryset = queryset.order_by("-score")

        if filters:
            queryset = queryset.apply_haystack_filters(filters)

        if top_k:
            queryset = queryset[:top_k]

        return queryset
