from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'), 
    path('faceDetect', views.faceDetect, name='faceDetect'),
    path('maskDetect', views.maskDetect, name='maskDetect'),
    path('mask_feed', views.mask_feed, name='mask_feed'),
    path('image_process', views.image_process, name='image_process'),
]