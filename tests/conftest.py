import django
import pytest
from django.conf import settings

def pytest_configure():
    settings.configure(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.postgresql",
                "NAME": "test_db",
                "USER": "postgres",
                "PASSWORD": "postgres",
                "HOST": "localhost",
                "PORT": "5432",
            }
        },
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "django.contrib.postgres",  # Add this for SearchVector support
            "tests.testapp",
            "django_haystack",  # Add this so Django finds our models
        ],
        USE_TZ=True,
    )
    
    # Initialize Django before running tests
    django.setup()
