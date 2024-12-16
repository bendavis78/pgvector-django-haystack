from django.db import models
from django_haystack.models import HaystackDocumentStoreModel
from pgvector.django import VectorField
from pgvector.django.functions import CosineDistance

class BasicDocument(HaystackDocumentStoreModel):
    id = models.CharField(max_length=100, primary_key=True)
    content = models.TextField(blank=True)
    meta = models.JSONField(default=dict)

class FullDocument(HaystackDocumentStoreModel):
    id = models.CharField(max_length=100, primary_key=True) 
    content = models.TextField(blank=True)
    embedding = VectorField(dimensions=3)
    meta = models.JSONField(default=dict)
    dataframe = models.JSONField(null=True)
    blob_data = models.BinaryField(null=True)
    blob_meta = models.JSONField(null=True)
    blob_mime_type = models.CharField(max_length=100, null=True)
    
    class HaystackOptions:
        vector_function = CosineDistance
