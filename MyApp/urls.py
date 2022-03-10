from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'), # this is the index page
    path('mask_feed', views.mask_feed, name='mask_feed'),
    path('image_process', views.image_process, name='image_process'),
]