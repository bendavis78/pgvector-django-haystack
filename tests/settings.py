DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": "test_pgvector_haystack",
        "USER": "postgres",
        "PASSWORD": "postgres",
        "HOST": "db",
        "PORT": "5432",
    }
}

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth", 
    "django.contrib.postgres",
    "tests.testapp",
    "django_haystack",
]

SECRET_KEY = "test-key-not-for-production"
USE_TZ = True
