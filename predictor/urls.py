from django.urls import path
from . import views

urlpatterns = [
    # Page Views
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('prediction/', views.prediction, name='prediction'),
    path('forecast/', views.forecast, name='forecast'),
    path('performance/', views.performance, name='performance'),
    path('dataset/', views.dataset, name='dataset'),
    path('about/', views.about, name='about'),
    path('problem-objectives/', views.problem_objectives, name='problem_objectives'),
    path('methodology/', views.methodology, name='methodology'),
    path('contact/', views.contact, name='contact'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('forgot-password/', views.forgot_password, name='forgot_password'),

    # API Endpoints
    path('api/states/', views.api_states, name='api_states'),
    path('api/districts/', views.api_districts, name='api_districts'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/forecast-trend/', views.api_forecast_trend, name='api_forecast_trend'),
    path('api/dashboard/', views.api_dashboard, name='api_dashboard'),
    path('api/forecast/', views.api_forecast, name='api_forecast'),
]