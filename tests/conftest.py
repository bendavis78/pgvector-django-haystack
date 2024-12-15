import os
import pytest
from django.conf import settings

@pytest.fixture(scope='session')
def django_db_setup():
    from django.conf import settings
    settings.DATABASES['default']['TEST'] = {
        'NAME': 'test_pgvector_haystack',
    }

def pytest_configure():
    """Configure Django settings for tests if not already configured"""
    if not settings.configured:
        settings.configure(
            DATABASES={
                'default': {
                    'ENGINE': 'django.db.backends.postgresql',
                    'NAME': os.environ.get('POSTGRES_DB', 'test_pgvector_haystack'),
                    'USER': os.environ.get('POSTGRES_USER', 'postgres'),
                    'PASSWORD': os.environ.get('POSTGRES_PASSWORD', 'postgres'),
                    'HOST': os.environ.get('POSTGRES_HOST', 'db'),
                    'TEST': {
                        'HOST': os.environ.get('POSTGRES_HOST', 'db'),
                    },
                    'PORT': os.environ.get('POSTGRES_PORT', '5432')
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
