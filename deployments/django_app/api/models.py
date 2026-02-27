from __future__ import annotations

from django.db import models


class Prediction(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    model_name = models.CharField(max_length=100, default="best_classifier")
    model_version = models.CharField(max_length=50, null=True, blank=True)

    # Store JSON (request + prediction)
    input_obj = models.JSONField()
    pred_obj = models.JSONField()

    latency_ms = models.IntegerField()

    def __str__(self) -> str:
        return f"Prediction#{self.id} {self.model_name} {self.created_at:%Y-%m-%d %H:%M:%S}"
