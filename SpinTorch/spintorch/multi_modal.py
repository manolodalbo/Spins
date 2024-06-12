import torch
import torch.nn as nn
class MModel(nn.module):
    def __init__(self,films):
        super(MModel,self).__init__()
        self.films = nn.ModuleList(films)
    def forward(self,x):
        outputs = []
        for film in nn.ModuleList(self.films):
            unnormalized_output = film(x)
            prob = unnormalized_output[:,1]/unnormalized_output.sum(dim=-1)
            outputs.append(prob)
        combined_outputs = torch.cat(outputs,dim=1)
        return combined_outputs