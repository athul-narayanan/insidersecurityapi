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
from utils.util import pre_process, group_sort, findunique
import re
from collections import defaultdict
from lime.lime_tabular import LimeTabularExplainer


base_feature_names = [
   'hour', 'num_to', 'num_cc', 'num_bcc', 'num_words',
   'has_attachment', 'O', 'C', 'E', 'A', 'N'
]


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

            input_data, device = pre_process(sorted_grouped_df)
            sequence_length = 5


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
            per_row_threshold = 1
            normal_sequences = []
            suspicious_sequences = []
            mal_rows=[]
            mal_row_indices = []
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

                    if malicious_row_indices ==0:
                        malicious_row_indices=[np.argmax(per_row_errors[i])]
                    mal_row_indices.append(malicious_row_indices)
                    malicious_row_indices = [x + i for x in malicious_row_indices]
                    mal_rows.append(sorted_grouped_df.iloc[malicious_row_indices])

            if len(mal_rows)!=0 :
                # Integrate Lime model for explainability
                sequence_length = input_data.size(1)
                input_dim = input_data.size(2)

                # Generate input for LIME
                normal_sequences = torch.tensor(np.array(normal_sequences), dtype=torch.float32).to(device)
                flattened_input = normal_sequences.reshape(normal_sequences.size(0), -1).cpu().numpy()
                background_data = flattened_input

                def predict_fn(X):
                    X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, sequence_length, input_dim).to(device)
                    with torch.no_grad():
                        reconstructed = model(X_tensor)
                        loss = torch.nn.functional.mse_loss(X_tensor, reconstructed, reduction='none')
                        reconstruction_error = loss.mean(dim=(1,2)).cpu().numpy()
                    return reconstruction_error.reshape(-1, 1)

                explainer = LimeTabularExplainer(background_data, mode="regression",
                                                feature_names=[f"t{t}_f{f}" for t in range(sequence_length) for f in range(input_dim)],
                                                discretize_continuous=True)

                top_features_list = []
                top_features_each_step = []
                for idx, seq_tensor in enumerate(suspicious_sequences):
                    top_features_each_step = []
                    instance = seq_tensor.cpu().numpy().flatten()
                    explanation = explainer.explain_instance(instance, predict_fn, num_features=50)

                    contrib = explanation.as_list()
                    
                    # Group the elements by step
                    timestep_contribs = defaultdict(list)
                    for feat, val in contrib:
                        match = re.search(r"t(\d+)_f(\d+)", feat)
                        if match:
                            t_index = int(match.group(1))
                            f_index = int(match.group(2))
                            if f_index < len(base_feature_names):
                                feature_name = base_feature_names[f_index]
                                timestep = f"t{t_index}"
                                timestep_contribs[timestep].append((feature_name, abs(val)))

                    # Ensure all time steps ate present to ensure that for each step we can idenfify
                    # The factor most contributing the error
                    for t in range(5):
                        timestep = f"t{t}"
                        if timestep not in timestep_contribs:
                            timestep_contribs[timestep] = []

                    # Find the top contributing feature
                    top_features_by_timestep = {}
                    for timestep, feats in timestep_contribs.items():
                        if feats:
                            top_feat, top_val = max(feats, key=lambda x: x[1])
                            top_features_by_timestep[timestep] = {
                                "feature": top_feat,
                                "contribution_percent": round(top_val * 100, 2)
                            }
                        else:
                            top_features_by_timestep[timestep] = {
                                "feature": None,
                                "contribution_percent": 0.0
                            }

                    top_features_each_step.append({
                        "sequence_index": idx,
                        "top_feature_per_timestep": top_features_by_timestep
                    })

                    for item in mal_row_indices[idx]:
                        top_features_list.append(top_features_each_step[0]["top_feature_per_timestep"][f't{item}']["feature"])

                    print(top_features_list) 
                # Define the output path
                filepath = str(time.time()) + "malicious_rows.xlsx" 
                output_path = os.path.join(settings.MEDIA_ROOT, filepath )


                # filter the malicious rows to have unique results

                
                malicious_df = pd.concat(mal_rows, axis=0).reset_index(drop=True)
                
               
                # Convert malicious records into JSON
                malicious_json = malicious_df.to_dict(orient="records")

                

                malicious_json , top_features_list = findunique(malicious_json,top_features_list)

                malicious_df = pd.DataFrame([malicious_json], index=[0]) 

                # Save the malicious records into excel
                malicious_df.to_excel(output_path, index=False)


                # Create the response to return
                return Response({
                    'malicious_rows': malicious_json,
                    'excel_path': filepath,
                    'features': top_features_list
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    'malicious_rows': {},
                    'excel_path': '',
                    'features': []
                }, status=status.HTTP_200_OK)
            

        return Response("Upload a file to proceed", status=status.HTTP_400_BAD_REQUEST)