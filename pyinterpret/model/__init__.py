"""The modules responsible for wrapping models through a common interface. Useful
If wed like to use the same api for models of different types (classifier v reg) or local
vs deployed."""

from .local import InMemoryModel
from .remote import DeployedModel

__all__ = ['InMemoryModel','DeployedModel']