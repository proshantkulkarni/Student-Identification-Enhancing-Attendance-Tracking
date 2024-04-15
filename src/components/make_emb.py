import cv2
import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from torchvision import transforms
from torchvision.io import read_image
import os
from glob import glob
from tqdm import tqdm
from facenet_pytorch import MTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS_PATH = "./weights/efficientnet_v2_s-dd5fe13b.pth"
IMAGE_PATH = "./student_photos"
EMB_PATH = "./student_emb"
os.makedirs(IMAGE_PATH, exist_ok=True)

cnn = efficientnet_v2_s()
cnn.load_state_dict(torch.load(WEIGHTS_PATH))
cnn.to(device)
cnn.eval()

detector = MTCNN(
    image_size=160,
    margin=35,
    post_process=False,
    select_largest=False,
    keep_all=True,
    device=device,
)

mypreprocess = transforms.Compose(
    [
        # transforms.CenterCrop(900),
        # transforms.Resize(2000),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


# Use cnn.features to extract features from face images
@torch.no_grad()
def get_embeddings(img):
    img = img.to(torch.float32) / 255.0 # [0, 1]
    batch = mypreprocess(img)  # preprocess(img).unsqueeze(0)
    batch = batch.to(device)
    features = cnn.features(batch)
    return features


images = sorted(glob(os.path.join(IMAGE_PATH, "*.jpeg")))
print(f"{len(images)} student images found...")

for im_path in tqdm(images):
    im = read_image(im_path)  # tensor
    face = detector(im.permute(1, 2, 0).detach().numpy())
    emb = get_embeddings(face)
    os.makedirs(EMB_PATH, exist_ok=True)
    filename = im_path.split(os.sep)[-1].split(".")[0] + ".pth"
    torch.save(emb, os.path.join(EMB_PATH, filename))

print("Face embedding created and saved successfully!")
