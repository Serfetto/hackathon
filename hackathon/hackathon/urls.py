from django.urls import path, include
from naseverecodit import views

urlpatterns = [
    path('', include('naseverecodit.urls')),
]

