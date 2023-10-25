
import gradio as gr 
import os
import torch 
import torchvision

from model impor create_effnetb2_model
from timeit import default_timer as timer 
from typing import Tuple, Dict

with open("class_names.txt", " r") as f:
    class_names = f.read().splitlines()

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=101)

effnetb2.load_state_dict(torch.load("09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth", map_location=torch.device("cpu")))


def predict(img) -> Tuple[Dict, float]:
    start_time = timer()

    img = effnetb2_transforms(img).unsqueeze(0)

    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)


    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in  range(len(class_names))}

    end_timer = timer()
    pred_time = round(end_timer-start_time, 4)

    return pred_labels_and_probs, pred_time


import gradio as gr

example_list = [
    ["examples/" + example] for example in os.listdir("examples")
]

title="FoodVision Big 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor model that predicts pizza, steak and sushi"
article= "Trained with 20 Percent of Data from FoodVision Mini"

demo = gr.Interface(fn=predict, 
                    inputs=gr.Image(type="pil"), 
                    outputs=[gr.Label(num_top_classes=5, label="Predictions"), 
                             gr.Number(label="Prediction Time (s)")], 
                    examples=example_list, 
                    title=title, 
                    description=description,
                    article=article)

demo.launch(debug=False, share=False)
