from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at", "model_name", "model_version", "latency_ms")
    search_fields = ("model_name", "model_version")
    list_filter = ("model_name",)
    ordering = ("-id",)