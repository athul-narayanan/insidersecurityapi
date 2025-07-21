from rest_framework import generics
from django.http import HttpResponse
from django.conf import settings
import os




class FileHandleView(generics.GenericAPIView):
    """
    This view is used to download and delete file
    """
    def get(self, request, file_name):
       filepath = os.path.join(settings.MEDIA_ROOT, file_name )
       data = None

       if not os.path.exists(filepath):
            return HttpResponse("File not found", status=404)
        
       with open(filepath, 'rb') as file:
            data = file.read()
            
       response = HttpResponse(data, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
       response['Content-Disposition'] = f'attachment; filename="{file_name}"'
       return response
