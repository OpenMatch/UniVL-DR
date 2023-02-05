import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class encoder_model(nn.Module):
    def __init__(self, encoder):
        super(encoder_model, self).__init__()
        self.encoder = encoder
        self.project_layer = nn.Linear(1024, 512)
        self.logit_scale = self.encoder.logit_scale

    def forward(self, text=None, image=None, captions=None):
        if text != None:
            embeddings = self.encoder(text=text, image=None)
        elif image != None:
            embeddings = self.encoder(image=image, text=None)
            if captions != None:
                cap_embeddings = self.encoder(text=captions, image=None)
                embeddings = torch.cat([embeddings, cap_embeddings], -1)
                embeddings = self.project_layer(embeddings)
        else:
            raise ("error!")
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings
