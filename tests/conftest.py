import os
import pytest
from django.conf import settings

def pytest_configure():
    """Configure Django settings for tests"""
    settings.configure(
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'NAME': os.environ.get('POSTGRES_DB', 'test_pgvector_haystack'),
                'USER': os.environ.get('POSTGRES_USER', 'postgres'),
                'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'postgres'),
                'HOST': os.environ.get('POSTGRES_HOST', 'localhost'),
                'PORT': os.environ.get('POSTGRES_PORT', '5432'),
            }
        },
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.postgres',
            'tests.testapp',
            'django_haystack',
        ],
        SECRET_KEY='test-key-not-for-production',
        USE_TZ=True,
    )
