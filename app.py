import streamlit as st
from PIL import Image
import torch
import requests
from transformers import YolosImageProcessor, YolosForObjectDetection

# Load the YOLOS model
model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')

def detect_objects(image):
    inputs = image_processor(images=image, return_tensors='pt')
    outputs = model(**inputs)

    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes

    # print results
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    
    objects = []
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        box = [round(i, 2) for i in box.tolist()]
        class_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 3)
        objects.append((class_name, confidence, box))

    return objects

# Streamlit app
st.title('Object Detection with YOLOS')
st.write('Enter the URL of an image for object detection.')

image_url = st.text_input('Image URL:')
if st.button('Detect Objects'):
    if image_url:
        try:
            response = requests.get(image_url, stream=True)
            image = Image.open(response.raw)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            objects = detect_objects(image)
            st.subheader('Detected Objects:')
            for obj in objects:
                st.write(f'Class: {obj[0]}, Confidence: {obj[1]}, Box: {obj[2]}')
        except:
            st.write('Error: Invalid Image URL or Unable to Fetch the Image.')
    else:
        st.write('Please enter the URL of an image.')

