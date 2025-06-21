import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import kornia

model_path = "classifier.pt"
num_classes = 11
model = Resnet18MoreThanRGB(num_classes)
model.image_size = 112
model = ModelMultiToBinaryWrapper(model, num_classes)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
transform = transforms.ToTensor()

test_labels = []
test_predictions_multi = []
test_predictions_binary = []
with torch.no_grad():
    for images, labels in rich.progress.track(test_dataloader):
        images = images.cuda()
        images = torchvision.transforms.functional.resize(images, (model.image_size, model.image_size))
        outputs_multi, outputs_binary = model(images, return_binary_logits=True)
        test_labels.append(labels.numpy())
        test_predictions_multi.append(outputs_multi.softmax(-1).cpu().numpy())
        test_predictions_binary.append(outputs_binary.softmax(-1).cpu().numpy())

test_labels = np.concatenate(test_labels)
test_predictions_multi = np.concatenate(test_predictions_multi)
test_predictions_binary = np.concatenate(test_predictions_binary)
