import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


class solver_inference_image(nn.Module):
    def __init__(self, student_model, image_size=224, device="cpu"):
        super().__init__()
        self.device = device
        self.student_model = student_model.to(device)
        self.student_model.eval()   # ✅ once, here

        self.img_size = (image_size, image_size)

        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
    def preprocess(self, image_rgb):
        image = Image.fromarray(image_rgb)
        return self.transform(image)
    @torch.no_grad()
    #def run(self, image_rgb):
    #    x = self.preprocess(image_rgb)
    #    x = x.unsqueeze(0).to(self.device)
    #    logits, _ = self.student_model(x)
    #    probs = torch.softmax(logits, dim=1)
    #    conf, pred = torch.max(probs, dim=1)
    #    return pred.item(), conf.item()
    def run(self, image_rgb):
        x = self.preprocess(image_rgb)
        x = x.unsqueeze(0).to(self.device)
        logits, _ = self.student_model(x)
        probs = logits.squeeze()
        # Softmax-like normalization
        #softmax_probs = torch.softmax(probs, dim=0)
        pred_idx = torch.argmax(probs).item()        # predicted class
        confidence = probs[pred_idx].item()
        return pred_idx, confidence
