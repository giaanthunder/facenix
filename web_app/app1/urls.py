from django.urls import path
from app1 import views

urlpatterns = [
   path('', views.app1, name='app1'),
]