"""
    serializer for file upload
"""

from rest_framework import serializers

class FileUploadSerializer(serializers.Serializer):
    """
        
    """
    file = serializers.FileField()