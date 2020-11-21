import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM( embed_size, hidden_size, num_layers = num_layers,dropout = 0,batch_first=True)
        self.hidden_size=hidden_size
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        captions=captions[:,:-1]
        captions=self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        out,h = self.lstm(inputs)
        out = self.fc(out)
        
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs=[]
        
        states = (torch.randn(1, inputs.shape[0], self.hidden_size).to(inputs.device),
              torch.randn(1, inputs.shape[0], self.hidden_size).to(inputs.device))
        
        while True:
            out,states= self.lstm(inputs,states)
            
            out = self.fc(out)
            out=out.squeeze(1)
            _, indices = torch.max(out, 1)
            outputs.append(indices.cpu().numpy()[0].item())
            if indices==1 or len(outputs)>=max_len:
                break
            inputs = self.embed(indices)   
            inputs = inputs.unsqueeze(1)
        return outputs
            