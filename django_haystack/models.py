import logging
from typing import Any, Dict

from django.core.exceptions import FieldDoesNotExist
from django.db import models
from django.db.models import Q
from django.db.models.base import ModelBase
from haystack.dataclasses.document import ByteStream, Document, SparseEmbedding
from pandas import read_json

log = logging.getLogger(__name__)


class HaystackModelFieldRequired(Exception):
    pass


class HaystackOptions:
    def __init__(self, declared_options: dict):
        self.model = None
        self.field_map = {
            "id": getattr(declared_options, "id_field", "id"),
            "content": getattr(declared_options, "content_field", "content"),
            "meta": getattr(declared_options, "meta_field", "meta"),
            "embedding": getattr(declared_options, "embedding_field", "embedding"),
            "sparse_embedding": getattr(
                declared_options, "sparse_embedding_field", "sparse_embedding"
            ),
            "dataframe": getattr(declared_options, "dataframe_field", "dataframe"),
            "blob_data": getattr(declared_options, "blob_data_field", "blob_data"),
            "blob_meta": getattr(declared_options, "blob_meta_field", "blob_meta"),
            "blob_mime_type": getattr(declared_options, "blob_mime_type_field", "blob_mime_type"),
            "score": getattr(declared_options, "score_field", "score"),
        }
        self.vector_function = getattr(declared_options, "vector_function", None)

    def __repr__(self):
        return f"<HaystackOptions for model {self.model.__name__}>"

    def contribute_to_class(self, cls, *_):
        required_fields = ("id", "meta")

        for field in required_fields:
            if field not in self.field_map:
                raise HaystackModelFieldRequired(
                    f"You must declare a field named '{self.field}' or map to another field using"
                    f"`{self.field_name}_field` in `HaystackOptions`."
                )

        self.model = cls
        cls._haystack = self

    def get_field(self, haystack_field_name: str) -> models.fields.Field | None:
        if haystack_field_name not in self.field_map:
            raise ValueError(f"Field {haystack_field_name} is not a valid haystack Document field")

        try:
            return self.model._meta.get_field(self.field_map[haystack_field_name])
        except FieldDoesNotExist:
            return None


class HaystackDocumentStoreQuerySet(models.QuerySet):
    def apply_haystack_filters(self, filters: Dict[str, Any]):
        """
        Filters a queryset based on haystack filters.
        """
        if "operator" not in filters and "conditions" not in filters:
            raise ValueError(
                "Invalid filter syntax. See "
                "https://docs.haystack.deepset.ai/docs/docs/metadata-filtering for details."
            )

        return self._apply_filters(filters)

    def _apply_filters(self, filter_spec: Dict[str, Any]) -> models.QuerySet:
        if "field" in filter_spec:
            return self.filter(self._parse_comparison(filter_spec))
        else:
            return self.filter(self._parse_logical(filter_spec))

    def _parse_comparison(self, filter_spec: Dict[str, Any]) -> Q:
        field = filter_spec["field"]
        operator = filter_spec["operator"]
        value = filter_spec["value"]

        if operator == "==":
            return Q(**{field: value})
        elif operator == "!=":
            return ~Q(**{field: value})
        elif operator == ">":
            return Q(**{f"{field}__gt": value})
        elif operator == ">=":
            return Q(**{f"{field}__gte": value})
        elif operator == "<":
            return Q(**{f"{field}__lt": value})
        elif operator == "<=":
            return Q(**{f"{field}__lte": value})
        elif operator == "in":
            return Q(**{f"{field}__in": value})
        elif operator == "not in":
            return ~Q(**{f"{field}__in": value})
        else:
            raise ValueError(f"Unsupported comparison operator: {operator}")

    def _parse_logical(self, filter_spec: Dict[str, Any]) -> Q:
        operator = filter_spec["operator"].upper()
        conditions = filter_spec["conditions"]

        if not isinstance(conditions, list) or not all(
            isinstance(cond, dict) for cond in conditions
        ):
            raise ValueError("'conditions' must be a list of filter dictionaries")

        if operator == "AND":
            q_objects = Q()
            for condition in conditions:
                q_objects &= (
                    self._parse_comparison(condition)
                    if "field" in condition
                    else self._parse_logical(condition)
                )
            return q_objects
        elif operator == "OR":
            q_objects = Q()
            for condition in conditions:
                q_objects |= (
                    self._parse_comparison(condition)
                    if "field" in condition
                    else self._parse_logical(condition)
                )
            return q_objects
        elif operator == "NOT":
            if len(conditions) != 1:
                raise ValueError("NOT operator must have exactly one condition")
            condition = conditions[0]
            return (
                ~self._parse_comparison(condition)
                if "field" in condition
                else ~self._parse_logical(condition)
            )
        else:
            raise ValueError(f"Unsupported logical operator: {operator}")


class HaystackDocumentMetaclass(ModelBase):
    """
    Metaclass to process HaystackOptions on model
    """

    def __new__(cls, name, bases, attrs):
        # Extract HaystackOptions if it exists
        declared_options = attrs.pop("HaystackOptions", {})
        new_model = super().__new__(cls, name, bases, attrs)

        options = HaystackOptions(declared_options)

        new_model.add_to_class("_haystack", options)

        return new_model


class HaystackDocumentStoreModel(models.Model, metaclass=HaystackDocumentMetaclass):
    """
    Base model for storing Haystack documents.
    """

    objects = HaystackDocumentStoreQuerySet.as_manager()

    class Meta:
        abstract = True

    def _get_haystack_field_value(self, haystack_field_name):
        field = self._haystack.get_field(haystack_field_name)
        return getattr(self, field.name, None) if field else None

    def to_haystack_document(self, **kwargs) -> Document:
        """
        Convert model instance to a Haystack Document.
        """
        # Construct a JSON object that can be passed to Document.from_dict()
        blob = None
        if blob_data := self._get_haystack_field_value("blob_data"):
            blob = ByteStream(
                data=blob_data,
                meta=self._get_haystack_field_value("blob_meta"),
                mime_type=self._get_haystack_field_value("blob_mime_type"),
            )

        if dataframe := self._get_haystack_field_value("dataframe"):
            dataframe = read_json(dataframe)

        if sparse_embedding := self._get_haystack_field_value("sparse_embedding"):
            sparse_embedding = SparseEmbedding(
                indices=sparse_embedding.indices,
                values=sparse_embedding.values,
            )

        attrs = dict(
            id=self._get_haystack_field_value("id"),
            content=self._get_haystack_field_value("content"),
            dataframe=dataframe,
            blob=blob,
            meta=self._get_haystack_field_value("meta"),
            score=self._get_haystack_field_value("score"),
            embedding=self._get_haystack_field_value("embedding"),
            sparse_embedding=self._get_haystack_field_value("sparse_embedding"),
        )
        attrs.update(kwargs)

        return Document(**attrs)

    @classmethod
    def from_haystack_document(cls, document: Document) -> models.Model:
        """
        Convert a Haystack Document to a model instance.
        """
        attrs = {
            "id": document.id,
            "content": document.content,
            "embedding": document.embedding,
            "dataframe": document.dataframe,
            "meta": document.meta,
            "sparse_embedding": document.sparse_embedding,
        }

        if document.blob:
            attrs["blob_data"] = document.blob.data
            attrs["blob_meta"] = document.blob.meta
            attrs["blob_mime_type"] = document.blob.mime_type

        kwargs = {}
        for doc_field, value in attrs.items():
            if field := cls._haystack.get_field(doc_field):
                kwargs[field.name] = value
            elif value is not None:
                log.warning(
                    "Cannot store value for `%s` as it is not defined in model '%s",
                    doc_field,
                    cls._meta.model_name,
                )

        return cls(**kwargs)
