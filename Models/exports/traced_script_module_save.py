#!/usr/bin/python3
import torch
import torchvision
from torch.export import export
from torchvision import transforms

import fx_parser
import calculate_ops_stats

import cv2
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser

import onnx
import onnx.inliner
from onnx import shape_inference

sys.path.append('..')

from inference.scene_seg_infer import SceneSegNetworkInfer
from model_components.scene_seg_network import SceneSegNetwork

##
## Example Usage: "python3 traced_script_module_save.py -p _checkpoint_file_.pth -i _test_image_.png -o _output_trace_file.pt"
##

def main(): 

    # Command line arguments
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path", help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath", help="path to input image which will be processed by SceneSeg")
    parser.add_argument("-o1", "--output_pt_trace_filepath", dest="output_pt_trace_filepath", help="path to *.pt output trace file generated")
    parser.add_argument("-o2", "--output_onnx_trace_filepath", dest="output_onnx_trace_filepath", help="path to *.onnx output trace file generated")
    parser.add_argument("-o3", "--output_layers_csv_filepath", dest="output_layers_csv_filepath", help="path to *.csv layers file generated")
    args = parser.parse_args() 

    # Saved model checkpoint path
    model_checkpoint_path = args.model_checkpoint_path

    # Checking devices (GPU vs CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'INFO: Using {device} for inference.')
        
    # Instantiate model, load to device and set to evaluation mode
    model = SceneSegNetwork()

    # Load the model weights etc.
    if(len(model_checkpoint_path) > 0):
        model.load_state_dict(torch.load \
            (model_checkpoint_path, weights_only=True, map_location=device))
    else:
        raise ValueError('ERROR: No path to checkpiont file provided in class initialization.')
    
    # Check the initialisation
    model = model.to(device)
    model = model.eval()

    # Image loader Helper
    image_loader = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # Reading input image
    input_image_filepath = args.input_image_filepath
    frame = cv2.imread(input_image_filepath, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((640, 320))

    # Prepare input image
    image_tensor = image_loader(image_pil)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Parse it through the fx parser.
    coverage, unparsed_nodes, _ = fx_parser.parse_model(model, input_shape=image_tensor.shape, output_file=args.output_layers_csv_filepath)

    # Coverage tells us the percentage of nodes in the graph being parsed correctly,
    # while unparsed nodes gives us a list of nodes traced as unrecognised.
    print(f"Nodes covered: {coverage * 100}")
    print(f"unparsed nodes: {unparsed_nodes}")
    print("INFO: FX Parser Trace CSV file generated successfully.")

    # Torch Export
    # Run and Trace the model with input image
    traced_script_module = torch.jit.trace(model, image_tensor)
    traced_script_module.save(args.output_pt_trace_filepath) 
    print("INFO: Torch Trace Export file generated successfully.")

    # ONNX export
    onnx_program = torch.onnx.dynamo_export(model, image_tensor)

    onnx_program.save(args.output_onnx_trace_filepath)
    onnx_model = onnx.load(args.output_onnx_trace_filepath)

    # Inline any functions
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx_model = onnx.inliner.inline_local_functions(onnx_model)

    onnx.save(onnx_model, args.output_onnx_trace_filepath)

    print("INFO: ONNX Export file generated successfully.")

if __name__ == '__main__':
    main()