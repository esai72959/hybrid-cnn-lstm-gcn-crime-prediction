import os
import sys
import threading
from django.apps import AppConfig


class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'

    def ready(self):
        # Run background warmup in a daemon thread so server starts instantly and ML models are pre-compiled
        if os.environ.get("RUN_MAIN") == "true" or "runserver" not in sys.argv:
            def _async_warmup():
                try:
                    from predictor.services.dataset_loader import DatasetLoader
                    from predictor.services.model_loader import ModelLoader
                    DatasetLoader().load_dataset()
                    ModelLoader().warm_up_models()
                except Exception as e:
                    import logging
                    logging.getLogger("predictor").warning("Async warmup encountered an issue: %s", e)

            thread = threading.Thread(target=_async_warmup, daemon=True, name="PredictorModelWarmup")
            thread.start()
