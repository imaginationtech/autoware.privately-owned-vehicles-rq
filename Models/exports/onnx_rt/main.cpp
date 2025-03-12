/*
**
FILE:   main.cpp
DESC:   C++ Deployment of SceneSeg Network
**
*/

#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "tvm-cpp-helper.hpp"
#include "cl-helper.hpp"

using namespace cv; 
using namespace std; 

namespace fs = std::filesystem;

using namespace tvm;
using namespace tvm::runtime;

/*
**
FUNC:   print_tensor_shape
DESC:   Prints shape of tensor
**
*/
std::string print_tensor_shape(const std::vector<std::int64_t>& vTensorShape) 
{
    std::stringstream stream("");

    for (unsigned int i = 0; i < vTensorShape.size() - 1; i++) 
    {
        stream << vTensorShape[i] << "x";
    }

    stream << vTensorShape[vTensorShape.size() - 1];
    
    return stream.str();
}

/*
**
FUNC:   main()
DESC:   Example program entry point
**
*/

int main(int argc, ORTCHAR_T* argv[]) 
{
    std::cout << std::endl << "\033[1;33m" << "==> Program Start:" << "\033[0m\n"  << std::endl;

    if (argc != 3) {
        std::cerr << "Usage: ./deploy_onnx_rt <onnx_model.onnx> <test_image.png>" << std::endl;
        return -1;
    }

    /*
    ************************************************************
    * ONNX RT to load and query the netowrk
    ************************************************************
    */

    // Save model file name
    std::basic_string<ORTCHAR_T> model_file = argv[1];

    // Check available execution platforms
    auto providers = Ort::GetAvailableProviders();
    std::cout << "INFO: Execution providers in this system:" << std::endl;
    
    for (auto provider : providers) {
        std::cout << "      - " << provider << std::endl;
    }
    std::cout << endl;

    // Create ORT Environment
    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");

    // Structures for options
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;

    // Single CPU thread only
    session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(1);

    // Optimization will take time and memory during startup
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Use CUDA
    if (false /* Use Cuda */)
    {
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
        cuda_options.arena_extend_strategy = 0;

        // May cause data race in some condition
        cuda_options.do_copy_in_default_stream = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options
    }

    // Create ONNX Session
    Ort::Session session = Ort::Session(env, model_file.c_str(), session_options);

    // Load the TVM Network
    DLDevice tvm_device {kDLOpenCL, 0};
    Module sceneseg = TVMHelper::LoadNetwork("./iter_140215_epoch_4_step_15999.so", "./iter_140215_epoch_4_step_15999.json", "./iter_140215_epoch_4_step_15999.params", tvm_device);

    // Input Shape Information
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::int64_t> input_shapes;

    std::cout << "INFO: Input Nodes (" << input_names.size() << "):" << std::endl;

    for (std::size_t i = 0; i < session.GetInputCount(); i++) 
    {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "      - " <<  input_names.at(i) << " : " << print_tensor_shape(input_shapes) << std::endl;
    }
    std::cout << endl;

    // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
    for (auto& s : input_shapes) 
    {
        if (s < 0) {
        s = 1;
        }
    }

    // Output Shape Information
    std::vector<std::string> output_names;
    std::cout << "INFO: Output Nodes (" << output_names.size() << "):" << std::endl;

    for (std::size_t i = 0; i < session.GetOutputCount(); i++) 
    {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "      - " << output_names.at(i) << " : " << print_tensor_shape(output_shapes) << std::endl;
    }
    std::cout << endl;

    // Assume model has 1 input node and 1 output node.
    assert(input_names.size() == 1 && output_names.size() == 1);

    // Create a single Ort tensor of random numbers
    auto input_shape = input_shapes;

    // Memory Initialisation - Used to allocate memory for input
    Ort::MemoryInfo memory_info{ nullptr };     

    try 
    {
        memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
    catch (Ort::Exception &oe) 
    {
        std::cout << "INFO: ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
        return -1;
    }

    // Input and Output Processing
	std::vector<const char*> inputNodeNames;
	std::vector<const char*> outputNodeNames;
	std::vector<int64_t> inputTensorShape;
	std::vector<int64_t> outputTensorShape;
	std::vector<int64_t> outputMaskTensorShape;

	auto temp_input_name0 = session.GetInputNameAllocated(0, allocator);
	inputNodeNames.push_back(temp_input_name0.get());

	Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
	auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();

	inputTensorShape = input_tensor_info.GetShape();

    std::vector<std::vector<int64_t>> input_node_dims;
    input_node_dims.push_back(inputTensorShape);

    // Onnx Runtime allowed input
    std::vector<Ort::Value> input_tensors;       

    /*
    ************************************************************
    * libTORCH and OpenCV to load and prepare the input image
    ************************************************************
    */

    // Load an input image
    cv::Mat frame = cv::imread(argv[2], cv::IMREAD_COLOR); 
    cv::Mat preImg;
    cvtColor(frame, preImg, cv::COLOR_BGR2RGB);

    // Resize
    cv::Mat img;
    resize(preImg, img, Size(640, 320));    // Fixed ratio from original network implementation

    // Convert input cv image to Tensor
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);

    // Convert to float and scale it 
    tensor_image = tensor_image.toType(c10::kFloat).div(255);

    // Transpose the image
    tensor_image = tensor_image.permute({ (2),(0),(1) });

    // Create a batch dimension for input
    tensor_image.unsqueeze_(0);

    // Normalise the input values
    std::vector<double> norm_mean = {0.485, 0.456, 0.406};
    std::vector<double> norm_std = {0.229, 0.224, 0.225};
    tensor_image = torch::data::transforms::Normalize<>(norm_mean, norm_std)(tensor_image);

    // Need to physically permute the input data from RGBRGBRGB to RRRGGGBBB... etc
    float fPhysicallyPermutedInputArray[1][3][320][640];

    std::cout << "INFO: Permuting Input Data:" << endl;
    for (int chan=0;chan<3;chan++)
    {
        for (int cols=0;cols<640;cols++)
        {
            for (int rows=0;rows<320;rows++)
            {
                fPhysicallyPermutedInputArray[0][chan][rows][cols] = tensor_image[0][chan][rows][cols].item<float>();
            }
        }
    }
    std::cout << "INFO: Done." << endl;

    /*
    ************************************************************
    * ONNX RT to run the network
    ************************************************************
    */
    
    // Create an NDArray for the input tensor
    DLDataType dtype = {kDLFloat, 32, 1}; // The inputs are float
    auto input_tensor = tvm::runtime::NDArray::Empty(input_shape, dtype, {kDLCPU, 0});
    input_tensor.CopyFromBytes(fPhysicallyPermutedInputArray, sizeof(fPhysicallyPermutedInputArray));

    std::cout << "INFO: Input_tensor shape: " << endl << "      - " << print_tensor_shape(input_shape) << std::endl << std::endl;

    std::cout << "INFO: Running model:" << std::endl;

    // Get the start time
    auto start = std::chrono::high_resolution_clock::now();

    auto output_tensors = TVMHelper::RunNetwork(input_names[0], sceneseg, output_names[0], input_tensor);

    // Get pointer to output tensor float values
    tvm::runtime::NDArray cpu_array = TVMHelper::CopyNDArrayToCPU(output_tensors[0]);

    size_t size = 0;
    float* floatarr = (float*) TVMHelper::GetNDArrayData(cpu_array, size);

    // Get the end time
    auto end = std::chrono::high_resolution_clock::now();

    // Output the duration in seconds
    std::chrono::duration<double> duration = end - start;
    std::cout << "INFO: Done. Time Taken: \t" << std::fixed << std::setprecision(5) << duration.count() << " seconds" << std::endl;

    // Copy ONNX Runtime Output Tensor Data to libTorch Tensor
    at::Tensor prediction = torch::from_blob(floatarr, { 1, 3, img.rows, img.cols}, at::kFloat);

    std::cout << "INFO: Forward compute path executed succssfully." << std::endl;

    /*
    ************************************************************
    * libTORCH to post process the output and save
    ************************************************************
    */

    // Post-process the output tensor
    prediction = prediction.squeeze(0).to(at::kCPU).detach();

    // Transpose back
    c10::IntArrayRef dims = {1, 2, 0};
    prediction = prediction.permute(dims);

    // Take max values across 2 dimensions
    auto final_output = std::get<1>(max(prediction,2,false));

    // Create an image to hold the segmentation mask
    cv::Mat vis_predict_object(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));

    // Process the tensor output and crate output segmentation mask
    for (int x=0;x<img.rows;x++) // rows
    {
        for (int y=0;y<img.cols;y++) // cols
        {
            int value = final_output[x][y].item<int>();
            
            if (value==0) 
            {
                // Background colour
                vis_predict_object.at<cv::Vec3b>(x,y)=cv::Vec3b(255,93,61);
            }
            if (value==1)
            {
                // Foreground colour
                vis_predict_object.at<cv::Vec3b>(x,y)=cv::Vec3b(145,28,255);
            }
            if (value==2)
            {
                // Background colour
                vis_predict_object.at<cv::Vec3b>(x,y)=cv::Vec3b(255,93,61);
            }
        }
    }

    std::cout << "INFO: Segmentation mask created succssfully." << std::endl;

    // Write out the segmentation mask to file
    cv::imwrite("output_seg_mask.jpg", vis_predict_object);

    // Transparency factor
    auto alpha = 0.5;

    // Create an image to hold the segmentation mask
    cv::Mat final_output_image(frame.rows, frame.cols, CV_8UC3, Scalar(0, 0, 0));

    // Resize
    cv::Mat out_img;
    resize(vis_predict_object, out_img, Size(frame.cols, frame.rows));    // Fixed ratio from original implementation

    // Alpha Blend of Segmentation mask onto original input image
    addWeighted( out_img, alpha, frame, 1-alpha, 0.0, final_output_image);

    std::cout << "INFO: Output image created succssfully." << std::endl;

    // Write out the segmentation mask to file
    cv::imwrite("output_image.jpg", final_output_image);

    std::cout << endl << "\033[1;33m" << "<== Program End" << "\033[0m\n" << std::endl << std::endl;

    // Done
    return 0;
}

/*
**
End of File
**
*/