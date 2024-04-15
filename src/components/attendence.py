import cv2
import torch
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from torchvision import transforms
import os
from glob import glob
from tqdm import tqdm
from facenet_pytorch import MTCNN
from pprint import pprint
import numpy as np

VID_PATH = "./class_recordings/2024-02-11.mp4"
WEIGHTS_PATH = "./weights/efficientnet_v2_s-dd5fe13b.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn = efficientnet_v2_s()
# cnn = efficientnet_v2_m()
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
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),# update these
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def get_frames(path):
    v_cap = cv2.VideoCapture(path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through video, taking a handful of frames to form a batch
    frames = []
    for i in tqdm(range(v_len)):
        # Load frame
        ret, frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    v_cap.release()
    return frames


def get_faces(frames):
    all_faces = []

    for frame in tqdm(frames):
        faces = detector(frame, return_prob=False)
        if faces is not None:
            # if len(faces.shape) == 3:
            #     faces = faces.unsqueeze(0)
            all_faces.extend(faces)
    return all_faces


@torch.no_grad()
def get_distance(a, b):
    # return torch.nn.functional.mse_loss(a, b)
    a = cnn.avgpool(a)
    b = cnn.avgpool(b)
    # return (a - b).norm().item()
    return torch.nn.functional.cosine_similarity(a.view(1, -1), b.view(1, -1)).item()


@torch.no_grad()
def match_faces(face, emb_path="./student_emb"):
    face = face.to(torch.float32) / 255.0
    face = mypreprocess(face)
    face = face.to(device)
    # print(face.min(), face.max())
    face_emb = cnn.features(face.unsqueeze(0))

    # k-nn algo
    database_embs = os.listdir(emb_path)
    min_dis = torch.inf
    stud_name = "unknown"
    for student in database_embs:
        path = os.path.join(emb_path, student)
        ref_emb = torch.load(path)

        ref_emb = ref_emb.to(device)
        face_emb = face_emb.to(device)

        dis = get_distance(ref_emb, face_emb)
        if dis < min_dis:
            min_dis = dis
            stud_name = student
    return stud_name, min_dis


def main():
    print("Getting frames...")
    all_frames = get_frames(VID_PATH)
    # from matplotlib import pyplot as plt

    print("Detecting faces...")
    all_faces = get_faces(
        all_frames
    )  # the queue consisting of all the faces in the video

    print("K-NN algo")
    present_stud = []
    for face in tqdm(all_faces):
        stud_name, dis = match_faces(face)
        present_stud.append((stud_name, dis))
    pprint(present_stud)
    # i = 1
    # for face, pred in zip(all_faces, present_stud):
    #     plt.subplot(1, 3, i)
    #     plt.imshow(face.permute(1, 2, 0).int())
    #     plt.title(pred)
    #     i += 1
    # plt.show()


if __name__ == "__main__":
    main()
