import os
import sys
import threading
from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        # Models and datasets load lazily on first prediction request,
        # ensuring sub-second container boot times on cloud deployments.
        pass
