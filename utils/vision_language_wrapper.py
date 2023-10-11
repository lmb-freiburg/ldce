import torch
from torch import nn

class VisionLanguageWrapper(nn.Module):
    def __init__(self, model, tokenizer, prompts) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts

        device = next(self.model.parameters()).device

        text = tokenizer(prompts)
        with torch.no_grad():
            self.text_features = model.encode_text(text.to(device))
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def forward(self, x):
        image_features = self.model.encode_image(x)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ self.text_features.T
        return logits