#%%
# Comment above is for Jupyter execution in VSCode
#! /usr/bin/env python3
import torch
from torchvision import transforms
import sys
sys.path.append('..')
from model_components.scene_seg_network import SceneSegNetwork

import tvm
from tvm.relay.op.contrib import imggc
from tvm.relay.op.contrib import imgnn

def tvm_load_pytorch(model, shape, tvm_device, use_imggc):
    input_name = "input0"
    shape_list = [(input_name, shape)]
    mod, params = tvm.relay.frontend.from_pytorch(model, shape_list)
    mod = tvm.relay.transform.InferType()(mod)

    if(use_imggc):
        mod = imggc.partition_for_imggc(mod, params)  ## Partition the Network for IMMGC
    else:
        mod = imgnn.partition_for_imgnn(mod, params)  ## Partition the Network for IMGNN

    ## Select the TVM runtime
    ## CPP is ok for when using Python
    runtime=tvm.relay.backend.Runtime('cpp')

    ## Select the TVM fallback target for the Ops that have not been partitioned
    fallback_target=tvm.target.Target("opencl", host=tvm.target.target.Target.from_device('cpu'))

    compile_options = { "verbose": "false" , "BVNC" : "35.4.1632.23" }

    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.imggc.options": compile_options}) as ctx:
        lib = tvm.relay.build(mod, target=[fallback_target], params=params, runtime=runtime)    

    module = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm_device))
    return module


class SceneSegNetworkInfer():
    def __init__(self, checkpoint_path = ''):

        # Image loader
        self.image_loader = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

        print(f'Using PowerVR for inference')
        self.device = torch.device('cpu')
            
        # Instantiate model, load to device and set to evaluation mode
        self.model = SceneSegNetwork()

        if(len(checkpoint_path) > 0):
            self.model.load_state_dict(torch.load \
                (checkpoint_path, weights_only=True, map_location=self.device))
        else:
            raise ValueError('No path to checkpiont file provided in class initialization')
        
        # FIXME: Is this required?
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        input_shape = [1, 3, 320, 640]
        input_data = torch.randn(input_shape)
        self.model = torch.jit.trace(self.model, input_data).eval()

        # FIXME: hardcoded shape
        self.tvm_device = tvm.device("opencl")
        self.module = tvm_load_pytorch(self.model, input_shape, self.tvm_device, True)

    def inference(self, image):

        width, height = image.size
        if(width != 640 or height != 320):
            raise ValueError('Incorrect input size - input image must have height of 320px and width of 640px')

        image_tensor = self.image_loader(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        self.module.set_input("input0", tvm.nd.array(image_tensor))
        self.module.run()
        self.tvm_device.sync()
        tvm_prediction = self.module.get_output(0)
        prediction = torch.from_dlpack(tvm_prediction.copyto(tvm.cpu(0)).to_dlpack())

        # Get output, find max class probability and convert to numpy array
        prediction = prediction.squeeze(0).cpu().detach()
        prediction = prediction.permute(1, 2, 0)
        _, output = torch.max(prediction, dim=2)
        output = output.numpy()

        return output
    