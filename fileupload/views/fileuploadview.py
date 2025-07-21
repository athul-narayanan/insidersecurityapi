from rest_framework import generics
from rest_framework import status
from rest_framework.response import Response
from django.conf import settings
from rest_framework import parsers
import pandas as pd
import torch
import torch.nn as nn
import os
import time
import numpy as np
from fileupload.serializer.fileuploadserializer import FileUploadSerializer
from utils.util import pre_process, group_sort
import shap


# Define the model same as model used for training
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoded, _ = self.decoder(hidden.repeat(x.size(1), 1, 1).permute(1, 0, 2))
        return decoded

class FileUploadView(generics.GenericAPIView):
    """
    This view is used to upload file
    """

    serializer_class = FileUploadSerializer
    parser_classes = (parsers.FormParser, parsers.MultiPartParser, parsers.FileUploadParser)


    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid(raise_exception=True):
            file = serializer.validated_data['file']
            file_ext = file.name.split('.')[-1].lower()


            if file_ext in ['csv']:
                df = pd.read_csv(file)
            elif file_ext in ['xls', 'xlsx']:
                df = pd.read_excel(file)

            sorted_grouped_df = group_sort(df)

            input_data, device = pre_process(df)
            sequence_length = 5

            print(input_data)

            input_dim = 11
            model = LSTMAutoencoder(input_dim=input_dim).to(device)
            # Load the model state from model.pt for evaluation
            model_path = os.path.join(settings.BASE_DIR, 'fileupload', 'static',  'model.pt')
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()


            with torch.no_grad():
                reconstructed = model(input_data)
                loss_fn = torch.nn.MSELoss(reduction='none')
                per_row_errors = loss_fn(reconstructed, input_data).mean(dim=2).cpu().numpy()
                reconstruction_error = loss_fn(reconstructed, input_data).mean(dim=(1, 2)).cpu().numpy()
                


            threshold = 0.2
            per_row_threshold = .5
            print("\nPrediction Results:")
            normal_sequences = []
            suspicious_sequences = []
            mal_rows=[]
            for i, err in enumerate(reconstruction_error):
                pred_status = "Suspicious" if err > threshold else "Normal"
                print(f"Sequence {i}: {pred_status} (Error: {err:.4f})")
                if pred_status == "Normal":
                    normal_sequences.append(input_data[i])
                else:
                    suspicious_sequences.append(input_data[i])
                    malicious_row_indices = [
                        j for j, val in enumerate(per_row_errors[i])
                        if val.item() > per_row_threshold  
                    ]

                    if len(malicious_row_indices) == 0:
                        malicious_row_indices=[np.argmax(per_row_errors[i])]
                    mal_rows.append(sorted_grouped_df.iloc[malicious_row_indices])
                    

            if len(mal_rows)!=0 :
                malicious_df = pd.concat(mal_rows, axis=0).reset_index(drop=True)

                # Define the output path
                filepath = str(time.time()) + "malicious_rows.xlsx" 
                output_path = os.path.join(settings.MEDIA_ROOT, filepath )

                # Save to Excel
                malicious_df.to_excel(output_path, index=False)

                # Optionally, convert to JSON
                malicious_json = malicious_df.to_dict(orient="records")

                # Add this in your view's return statement:
                return Response({
                    'malicious_rows': malicious_json,
                    'excel_path': filepath
                }, status=status.HTTP_200_OK)
            

        return Response("Upload a file to proceed", status=status.HTTP_400_BAD_REQUEST)