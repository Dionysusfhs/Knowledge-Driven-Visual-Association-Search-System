import os
import csv
import random
import PIL.Image
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import display
from flask import Flask, render_template, request
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def load_blip(pretrained, device):
    model = BlipForConditionalGeneration.from_pretrained(pretrained, torch_dtype=torch.float16).to(device)
    processor = BlipProcessor.from_pretrained(pretrained)
    
    return model, processor

def load_clip(pretrained, device):
    model = CLIPModel.from_pretrained(pretrained).to(device)
    processor = CLIPProcessor.from_pretrained(pretrained)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained)
    return model, processor, tokenizer

# 载入模型和特征
clip_model, clip_processor, clip_tokenizer = load_clip("E://UESTC//Junior(Autumn)//KRR//project//fake网页//clip", 'cuda')
blip_model, blip_processor = load_blip("E://UESTC//Junior(Autumn)//KRR//project//fake网页//blip", 'cuda')
feature = np.load('feature.npy', allow_pickle=True).item()
img_feature = feature['img_feature']
text_feature = feature['text_feature']
img_id = feature['img_id']

def search_by_img(clip_model, clip_processor, image, data, device = 'cuda'):    
    img = clip_processor(text=None, images = image, return_tensors="pt")["pixel_values"].to(device)
    img_embedding = clip_model.get_image_features(img)
    img_np = img_embedding.cpu().detach().numpy()
    
    img_sim = cosine_similarity(data['img_feature'], img_np).reshape(-1)
#     print(img_sim)
    k = 4
#     print(img_sim.max())
    img_id = indices_of_top_k = np.argsort(img_sim)[-k:][::-1]

    
    return img_id

def search_related_images(image):
    _id = search_by_img(clip_model, clip_processor, image, feature, device='cuda')
    img_id_unicode = np.char.decode(img_id)
    image_files = img_id_unicode[_id]
    return image_files

def load_image(image_path):
    return Image.open(image_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        print(file)
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            # Save the uploaded image to a temporary folder
            image_path = file.filename
            file.save(image_path)
            print(image_path)
            # Load the uploaded image
            uploaded_image = load_image(image_path)

            # Perform image search
            related_images = search_related_images(uploaded_image)
            related_images1 = []
            for i in related_images:
                print(type(i))
                i = "flickr30k-images/" + i
                related_images1.append(i)
            print(image_path)
            # related_images_test = ["flickr30k-images/7300624.jpg" , "flickr30k-images/7300628.jpg"]    
            # Display the uploaded image and related images
            return render_template('index.html', uploaded_image=image_path, related_images=related_images1)

    return render_template('index.html')

if __name__ == '__main__':
    #app.static_folder = os.path.abspath("E://UESTC//Junior(Autumn)//KRR//project//flickr 30k//flickr30k-images")
    app.run(debug=True)