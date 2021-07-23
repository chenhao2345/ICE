from __future__ import absolute_import

from .contrastive import ViewContrastiveLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .triplet import TripletLoss, SoftTripletLoss

__all__ = [
    'CrossEntropyLabelSmooth',
    'SoftEntropy',
    'TripletLoss',
    'SoftTripletLoss',
    'ViewContrastiveLoss']
