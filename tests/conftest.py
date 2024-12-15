import os
import pytest
from django.conf import settings

@pytest.fixture(scope='session')
def django_db_setup():
    from django.conf import settings
    settings.DATABASES['default']['TEST'] = {
        'NAME': 'test_pgvector_haystack',
    }

