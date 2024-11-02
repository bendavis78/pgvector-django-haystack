# django-haystack

`django-haystack` integrates with the [Haystack] LLM orchestration framework. This project provides:

- A Django model-backed [Document store]
- A default QuerySet with support for the Haystack [metadata filter spec]
- [Retrievers] for embedding-based and keyword-based retrieval

## Usage

### Document store model

First, define a model for the document store. At a minimum, you must define an
`id` field and an `meta` field. It is highly recommended that you define indexes
for your use-case. Refer to the [pgvector-python] documentation for details.

The following example shows a simple use case for storing documents for a 
"cosine similarity" search strategy.

```python
from django_haystack.models import HaystackDocumentStoreModel
from pgvector.django import HnswIndex, VectorField
from pgvector.django.functions import CosineDistance

class MyDocument(HaystackDocumentStoreModel):
    id = models.CharField(unique=True, primary_key=True)
    content = models.TextField(blank=True) 
    embedding = VectorField(dimensions=1024, null=True, blank=True)
    meta = models.JSONField(default=dict)

    class Meta:
        indexes = [
            HnswIndex(
                name="embedding_hnsw_index",
                fields=["embedding"],
                opclasses=["vector_cosine_ops"],
                m=16,
                ef_construction=64,
            ),
        ]

    class HaystackOptions:
        vector_function = CosineDistance
```

A [haystack Document] can be converted to a model instance using the
`from_haystack_document` class method:

```
from haystack.dataclasses import Document

doc = Document(content="Hello World!")
db_doc = MyDocument.from_haystack_document(doc)
db_doc.save()
```

Conversly, a `HaystackDocumentStoreModel` can be converted to a 
[haystack Document] instance using the `.to_haystack_document()` model method:

```
db_doc = MyDocument.objects.get(id=some_document_id)
haystack_doc = db_doc.to_haystack_document()
```

By default, the following fields are mapped from your model to the haystack [Document]:

- id
- content
- embedding
- meta
- dataframe
- blob_data, blob_meta, blob_mime_type (used to construct Document.blob)

You can re-map any of these fields by declaring `{name}_field` in the
`HaystackOptions` class in your model definition:

```
class MyDocument(HaystackDocumentStoreModel):
    id = fields.UUIDField(primary_key=True)
    haystack_id = fields.CharField(unique=True)
    embedding = VectorField()

    class HaystackOptions:
        id_field = "haystack_id"
```

In the above example, the Document `id` field will be mapped to the `haystack_id`
field on the model so that it does not interfere with the existing `id` field.

### Indexes & vector functions

You will likely want to use an index on your document store depending on your
use case. More info on defining indexes can be found in the [pgvector-python]
documentation.

If you use a HNSW index, you will need to define an `opclasses` argument which
will contain the vector function used for the index. During retrieval you will
need use the appropriate vector function class from [pgvector.django.functions].
For example, if your index uses the `vector_cosine_ops` in the `opclasses`
argument, your retrievals will need to use the `CosineDistance` function. See
the "Retrievers" section below for more info.

You can declare the default vector function used for lookups on the model's
`HaystackOptions` class:

```
class MyDocument(HaystackDocumentStoreModel):
    id = fields.UUIDField(primary_key=True)
    haystack_id = fields.CharField(unique=True)
    embedding = VectorField(dimensions=1024)

    class Meta:
        indexes = [
            HnswIndex(
                name='my_index',
                fields=['embedding']
                m=16,
                ef_construction=64,
                opclasses=['vector_cosine_ops']
            )

    class HaystackOptions:
        id_field = "haystack_id"
        vector_function = CosineDistance    
```

### Document store
The `DjangoModelDocumentStore` follows the haystack [DocumentStore] spec.

You can save Haystack documents to the document store by calling
`write_documents()`:

```
from haystack.dataclasses import Document
from django_haystack.document_store import DjangoModelDocumentStore

documents = [
    Document("The quick brown fox jumped over the lazy dog"),
    Document("The five boxing wizards jump quickly."),
]
document_store = DjangoModelDocumentStore(model=MyDocument)
document_store.write_documents([Document])
```

### Retrievers

Two retriever components are included: a `DjangoModelEmbeddingRetriever` and a
`DjangoModelKeywordRetriever`. These can be used in a [Pipeline] to retrieve
documents from the document store. A retriever takes 

```
from django_haystack.document_store import DjangoModelDocumentStore
from django_haystack.retrievers import DjangoModelEmbeddingRetriever

query_embedder = OpenAITextEmbedder(dimensions=1024, model="gpt-4o-mini")
document_store = DjangoModelDocumentStore(model=MyDocument)
retriever = DjangoModelEmbeddingRetriever(document_store=document_store)
pipeline = Pipeline()
pipeline.add_component("query_embedder", query_embedder)
pipeline.add_component("my_retriever", retriever)
pipeline.connect("query_embedder.embedding", "retriever.embedding")
```

The retriever will rank and score documents in the `score` field. The ordering
used depends on the vector function used in the retrieval. Using `L1Distance` or
`L2Distance` will result in a queryset ranked by `score` in ascending order.
All other vector functions will result in descending `score` rank order.

By default, the DjangoModelEmbeddingRetriever will use the `vector_function`
declared on the model's `HaystackOptions` class. You can also pass a
`vector_function` to the retriever's init arguments or at pipeline runtime via
the `vector_function` argument. In any case, the vector function must be a class
from [pgvector.django.functions]:

```
retriever = DjangoModelEmbeddingRetriever(
    document_store=document_store,
    vector_function=CosineDistance
)
```

### Metadata filters

A `HaystackDocumentStoreModel` will use the `HaystackDocumentQueryStore`
queryset by default. The queryset supports metadata filtering according to
Haystack's [metadata filter spec]. Both retrievers will get a queryset from the
`document_store` and apply any filters defined in the `filters` argument.

For more information on filters, see the 
[Haystack documentation][metadata filter spec].


[Haystack]: https://github.com/deepset-ai/haystack
[Document]: https://docs.haystack.deepset.ai/docs/data-classes#document 
[Pipeline]: https://docs.haystack.deepset.ai/docs/pipelines
[DocumentStore]: https://docs.haystack.deepset.ai/docs/document-store
[metadata filter spec]: https://docs.haystack.deepset.ai/docs/metadata-filtering
[pgvector-python]: https://github.com/pgvector/pgvector-python?tab=readme-ov-file#django
[haystack Document]: https://docs.haystack.deepset.ai/docs/data-classes#document
[pgvector.django.functions]: https://github.com/pgvector/pgvector-python/blob/master/pgvector/django/functions.py
