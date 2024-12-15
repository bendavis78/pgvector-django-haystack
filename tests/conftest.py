import pytest
from django.apps import apps
from django.db import connection
from django.db.models.signals import pre_migrate


def _pre_migration(sender, app_config, **kwargs):
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")


@pytest.fixture(autouse=True, scope="session")
def django_test_environment(django_test_environment):
    pre_migrate.connect(_pre_migration, sender=apps.get_app_config("testapp"))
