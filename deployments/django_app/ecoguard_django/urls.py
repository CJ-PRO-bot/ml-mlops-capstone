"""
URL configuration for ecoguard_django project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.http import JsonResponse
from django.urls import path, include

def root(request):
    return JsonResponse({"message": "EcoGuard Django API running. Try /api/health"})

urlpatterns = [
    path("", root),                     # GET /
    path("admin/", admin.site.urls),    # Admin
    path("api/", include("api.urls")),  # Your API endpoints
]