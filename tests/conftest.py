import os
import pytest
from django.conf import settings
from django.core.management import call_command

@pytest.fixture(autouse=True, scope="session")
def apply_migrations(django_db_setup, django_db_blocker):
    with django_db_blocker.unblock():
        call_command("migrate")


