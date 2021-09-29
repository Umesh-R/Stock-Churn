from django.urls import path
from . import views


urlpatterns = [
    path('', views.home,name='Home'),
    path('home', views.home,name='Home'),
    path('inputStocks',views.inputStocks,name='inputStocks'),
    path('processStocks',views.processStocks,name='processStocks')
]