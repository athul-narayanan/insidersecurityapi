"""
defines url mapping for the user API
"""

from django.urls import path
from fileupload.views.fileuploadview import FileUploadView
from fileupload.views.filedownloadview import FileHandleView
urlpatterns = [
    path('', FileUploadView.as_view()),
    path('/<str:file_name>', FileHandleView.as_view()),
]