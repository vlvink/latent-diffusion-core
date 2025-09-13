import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(CLIPTextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

    def forward(self, texts):
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**tokens)
        return outputs.last_hidden_state