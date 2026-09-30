# Importing the modules registers every model class for encoders.base.load_model.
from . import image, timeseries, video  # noqa: F401
from .base import LatentAutoencoder, load_model, save_model

__all__ = ["LatentAutoencoder", "load_model", "save_model"]
