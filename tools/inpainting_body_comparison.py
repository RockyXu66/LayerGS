import os
import sys
# Exclude local ImageReward folder to use the installed package
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if not p.startswith(script_dir)]

import torch
import ImageReward as RM
from pathlib import Path

# import clip
from tabulate import tabulate
from PIL import Image
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_name', type=str, default='', help='subject name')
    parser.add_argument('--gs_folder_path', type=str, default='', help='gs folder path')
    parser.add_argument('--text_prompt', type=str, default='', help='text prompt')
    args = parser.parse_args()

    if args.text_prompt == '':
        prompt = "a man wearing pants and a white tank top with black background"
    else:
        prompt = args.text_prompt
    img_prefix = "assets/compared_images"

    dataset_name = '4d-dress'
    subject_name = args.subject_name
    gs_images_folder = Path(args.gs_folder_path)
    num_imgs = len(os.listdir(gs_images_folder))
    model = RM.load("ImageReward-v1.0")
    win_cnt = [0, 0]
    gala_IR_score_list = []
    gala_wo_scan_IR_score_list = []
    mlgs_IR_score_list = []
    with torch.no_grad():
        # for y_angle in range(0, 360//20):
        # for y_angle in range(72):
        for y_angle in range(num_imgs):
            img_list = [
                # str(gala_images_folder / f'{y_angle:08d}.png'),
                # str(gala_wo_scan_images_folder / f'{y_angle:08d}.png'),
                str(gs_images_folder / f'{y_angle:08d}.png'),
            ]
            ranking, rewards = model.inference_rank(prompt, img_list)
            # Print the result
            # print("\nPreference predictions:\n")
            for index in range(len(img_list)):
                score = model.score(prompt, img_list[index])
                mlgs_IR_score_list.append(score)


    # Calculate clip
    import torch
    from PIL import Image
    import open_clip


    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    text = tokenizer([prompt])

    gala_clip_score_list = []
    gala_wo_scan_clip_score_list = []
    mlgs_clip_score_list = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # for y_angle in range(0, 360//20):
        # for y_angle in range(72):
        for y_angle in range(num_imgs):
            img_list = [
                # str(gala_images_folder / f'{y_angle:08d}.png'),
                # str(gala_wo_scan_images_folder / f'{y_angle:08d}.png'),
                str(gs_images_folder / f'{y_angle:08d}.png'),
            ]
            for index in range(len(img_list)):
                image = preprocess(Image.open(img_list[index])).unsqueeze(0)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                cos_sim = torch.matmul(image_features, text_features.T).item()
                clip_score = max(100 * cos_sim, 0)
                mlgs_clip_score_list.append(clip_score)
            # print()

    data = {
        # 'GALA w/ scan': [np.average(gala_clip_score_list), np.average(gala_IR_score_list)],
        # 'GALA w/o scan': [np.average(gala_wo_scan_clip_score_list), np.average(gala_wo_scan_IR_score_list)],
        'Ours w/o scan': [np.average(mlgs_clip_score_list), np.average(mlgs_IR_score_list)],
    }
    table_data = [[key] + value for key, value in data.items()]
    print(tabulate(table_data, headers=[subject_name, 'CLIP Score \u2191', 'ImageReward \u2191'], tablefmt='grid', floatfmt=["", ".2f", ".3f"]))