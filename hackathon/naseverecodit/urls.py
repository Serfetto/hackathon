from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.result, name='result'),
    path('record_audio', views.record_audio, name='record_audio'),
]