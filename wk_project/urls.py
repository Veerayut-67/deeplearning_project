from django.contrib import admin
from django.urls import path
from dashboard.views import index, line_webhook,clear_data,export_csv

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('line/callback/', line_webhook, name='line_webhook'),
    path('clear-data/', clear_data, name='clear_data'),
    path('export-csv/', export_csv, name='export_csv'),
]