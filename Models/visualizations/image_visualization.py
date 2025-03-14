#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
sys.path.append('..')
from inference.scene_seg_infer_powervr import SceneSegNetworkInfer


def make_visualization(prediction):
    shape = prediction.shape
  
    row = shape[0]
    col = shape[1]
    vis_predict_object = np.zeros((row, col, 3), dtype = "uint8")

    background_objects_colour = (255, 93, 61)
    foreground_objects_colour = (145, 28, 255)

    # Extracting predicted classes and assigning to colourmap
    for x in range(row):
        for y in range(col):
            if(prediction[x,y].item() == 0):
                vis_predict_object[x,y] = background_objects_colour
            elif(prediction[x,y].item() == 1):
                vis_predict_object[x,y] = foreground_objects_colour
            elif(prediction[x,y].item() == 2):
                vis_predict_object[x,y] = background_objects_colour
               
    return vis_predict_object

def main(): 

    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath", help="path to input image which will be processed by SceneSeg")
    args = parser.parse_args() 

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path
    model = SceneSegNetworkInfer(checkpoint_path=model_checkpoint_path)
  
    # Transparency factor
    alpha = 0.5

    # Reading input image
    input_image_filepath = args.input_image_filepath
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Run inference and create visualization
    prediction = model.inference(image_pil)
    vis_obj = make_visualization(prediction)

    # Resize and display visualization
    vis_obj = cv2.resize(vis_obj, (frame.shape[1], frame.shape[0]))
    image_vis_obj = cv2.addWeighted(vis_obj, alpha, frame, 1 - alpha, 0)
    cv2.imshow('Prediction Objects', image_vis_obj)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
# %%