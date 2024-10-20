# 🚗 Autonomous Vehicles Projects

Welcome to my Autonomous Vehicles Project repository! This repository contains resources for fine-tuning a **DETR** model and performing inference on real-world road footage, along with a small-scaled literature review on **Out-of-Distribution (OOD)** detection and current solutions for **collision-avoidance decision-making** in autonomous vehicles (AV).

_Please make sure you read this readme very carefully since it contains a lot of information that I have tried my best to research and solve the problem as much as possible._ 

## 📝 Problem Description:
_Despite the popularity of autonomous vehicles, the robustness of these vehicles degrades when operating in the rural areas or new environments. To address this challenge, your task is to build an AI system that has the following capabilities during the vehicle navigation:_ 

_1.  Detect known objects in the video 
2.  Detect novel objects in the video
3. Make decisions to avoid collisions on the road._

_Note that the second and third capabilities are crucial for any autonomous vehicle to improve its robustness._

____

## 📂 Project Structure
1. **Detecting Known Objects with DETR**
2. **Current OOD Objects Detection Universial Research Progress**
3. **A Study of Distance Measurement System**
4. **Proposed pipeline for AV**
5. **Future works**

____

Sure! Here’s an updated version of your README with icons added for visual appeal. I've used emojis to represent different sections, making it more engaging. You can replace them with icons from a library or image files if you prefer.

----------

# 🚗 Autonomous Vehicles Projects

Welcome to my Autonomous Vehicles Project repository! This repository contains resources for fine-tuning a **DETR** model and performing inference on real-world road footage, along with a small-scaled literature review on **Out-of-Distribution (OOD)** detection and current solutions for **collision-avoidance decision-making** in autonomous vehicles (AV).

## 📝 Problem Description

_Despite the popularity of autonomous vehicles, their robustness degrades when operating in rural areas or new environments. To address this challenge, your task is to build an AI system that has the following capabilities during vehicle navigation:_

1.  **Detect known objects** in the video.
2.  **Detect novel objects** in the video.
3.  **Make decisions** to avoid collisions on the road.

_Note: The second and third capabilities are crucial for any autonomous vehicle to improve its robustness._

----------

## 📂 Project Structure

1.  **Detecting Known Objects with DETR**
2.  **Current OOD Object Detection Universal Research Progress**
3.  **A Study of Distance Measurement Systems**
4.  **Proposed Pipeline for AV**
5.  **Future Works**

----------

## 🔍 Part 1: Detecting Known Objects with DETR
For this task, a fine-tune notebook was created for demonstation on how one can enhance the performance of pre-trained model by using dataset related specific to the task of road navigation.

_**Note**: Because of the time constraint, the fine-tuned model for the full dataset's not avaiable yet, see **'Future Works'** for more details. The fine-tune cde can be found in **'finetune.ipynb**_

___

## 📊 Dataset: Udacity Self Driving Car Dataset
- [Self Driving Car Dataset](https://public.roboflow.com/object-detection/self-driving-car)
- Description: The dataset contains 97,942 labels across 11 classes and 15,000 images. There are 1,720 null examples (images with no labels). All images are 1920x1200 (download size ~3.1 GB). Annotations have been hand-checked for accuracy by Roboflow.

![Class Balance](https://i.imgur.com/bOFkueI.pnghttps://)

**For quick fine-tuning demonstration, the dataset has been dowmsampled to the size 2000 images only.** The data also downloaded with COCO format annotation to fit with DETR's desired input.

### 🗃️ Additional Useful Datasets for AV Object Detection/Segmentation Training:

 - [Waymo Open Dataset](https://waymo.com/open/)
 - [A2D2 Dataset](https://www.a2d2.audi/a2d2/en.html)
 - [Apollo Scape Dataset](https://apolloscape.auto/)
 - [Argoverse Dataset](https://www.argoverse.org/)
 - [Berkeley DeepDrive Dataset](https://www.bdd100k.com/)
 - [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
 - [Comma2k19 Dataset](https://github.com/commaai/comma2k19)
 - [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/)
 - [LeddarTech PixSet Dataset](https://leddartech.com/solutions/leddar-pixset-dataset/)
 - [Level 5 Open Data](https://self-driving.lyft.com/level5/data/)
 - [nuScenes Dataset](https://www.nuscenes.org/)
 - [Oxford Radar RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/)
 - [PandaSet](https://pandaset.org/)

___

## 📐 Pre-trained Architecture: DEtection TRansformer (DETR)
- Proposed in the paper '[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)', DETR consists of a convolutional backbone followed by an encoder-decoder Transformer which can be trained end-to-end for object detection.
- For the fine-tune demontration, [DETR-resnet-50](https://huggingface.co/facebook/detr-resnet-50) with Resnet50 as the backbone architecture were chosen.

[DETR overview](https://huggingface.co/docs/transformers/en/model_doc/detr)

**Alternative choice**: YOLO's variants can also be ultilized for this task if one wants low computational cost and faster inference ([Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)). Pre-trained models dedicated to image segmentation can also be choosen but one must be assure about their resources since image segmentation are more costly to run.

___
## ⚙️ Fine-tune Pipeline Overview

 1. **Data preparation**: 
 - Map all images to their corresponding annotation using COCO format
 - Divide the dataset into train-val-test ratio of 0.8-0.1-0.1
 - Pass all sets to DETR processor to apply required transformations, for the train set apply extra random augmentation to introduce new data to each training epoch (random horizontal flip, random brightess)

2. **Model initialization**:
- Get pre-trained checkpoint HuggingFace's Models Hub and freeze its parameters
- Define LoRA configuration to aid fine-tune process.
_Note: To choose the layer for applying LoRA, I had to go in and look at the model architedcture and count the number of trainable parameters for each layer. As stated before, DETR consists of a CNN backbone and attention layers. For quick dementration, I decided to only choose the very last output projection layer, the labels classifer layer and the bounding boxes predictor layer. Anyone with extra resource and time should consider to fine-tune all of the model or the encoder-decoder layers with the full dataset for better performance. One could also increase the rank of the LoRA matrix to enhance fine-tuning experience._

3. **Model Training**:
- Define parameter required for training: numbers of epoch, batch size, learning rate, optimizer, learning rate scheduler
_Note: For fine-tuning demonstration, I only choose the universial values that I usually see in others people works. Ideally, one should perform hyperparameter tuning to get the best set of parameter to optimize model's performance_
- Create training loop that get a batch of dataset, feed into the model, get the output losses, reset optimizer's gradients, perform backpropagation and update the model's parameters. After each epoch, the learning rate is also updated with the sheduler. For validation loop, no backward pass is needed.
_Note: Access to GPU/TPU should improve the the computational and time cost, for this project I use my laptop's GPU but it still wasn't enough to train more parameters/larger dataset._

**The fine-tune notebook does not contain all the steps/techniques needed for a full model development process. See 'Future Works' for more details on what should be implemented in the future.**

**The notebook `inference.ipynb` can be used to simulate real-scenario detection using any video in the `video` directory, keep in mind that the model was used in the notebook was the pre-trained model not the fine-tuned one using the above dataset.**

___
## 📊 Part 2: Current OOD Objects Detection Universal Research Progress

The task of detect novel objects in an image/video can be translated into the problem of **Out-of-distribution (OOD) Object Detection**. _From my understanding_, OOD Detection is basically a task where the model has to regconize an object it have never seen before (ie. the unknown object does not belong to a class the model is trained on) and avoid treating the unknown object as part of the background. The field of OOD research encompasses several key areas:

-   **🛠️ OOD Detection**:  aims to identify when a model is presented with inputs that deviate from its training distribution. This allows systems to flag unusual cases for human review or fallback strategies.
-  **🔒 OOD Robustness**:  focuses on maintaining reliable performance even when test data differs from training data in important ways.
-   **🌍 OOD Generalization**:  seeks to develop models and techniques that can successfully extrapolate to novel domains and tasks without requiring retraining.

_I also found a GitHub repository contains collected articles, benchmarks, papers,... related to OOD:_ [OOD Machine Learning: Detection, Robustness, and Generalization](https://github.com/huytransformer/Awesome-Out-Of-Distribution-Detection)

_Unfortunately, I couldn't found any benchmarks in object detection/segmentation specificly. Because of the time contraint, I was only able to look through the OOD Detection part but not OOD Robustness, OOD Generalization and OOD Everything else part of the repository. But here are my key findings in some of the most recent papers about OOD Detection in object detection/segmentation that I found. I also chose papers with public implemented code._

___

### 📄 1. [SAFE: Sensitivity-Aware Features for Out-of-Distribution Object Detection](https://openaccess.thecvf.com/content/ICCV2023/papers/Wilson_SAFE_Sensitivity-Aware_Features_for_Out-of-Distribution_Object_Detection_ICCV_2023_paper.pdf)  [[Code]](https://github.com/SamWilso/SAFE_Official)

SAFE introduces a new approach to detect OOD objects using **Sensitivity-Aware Features** extracted from residual convolutional layers with batch normalization in object detection models. It is designed to identify objects that fall outside the training distribution without retraining the base object detection model, leveraging features sensitive to input variations. Here is how it was implemented:
-   **Base Network Architecture**: They implement the Faster RCNN detector with either a ResNet-50 or Reg NetX4.0 backbone using the Detectron2 library. For a fair comparison, they report the results of SAFE and VOS using both the ResNet-50 and RegnetX4.0 backbones to ensure that the differing on-task performance.
-   **Feature Extraction**: During feature extraction, hooks are applied to the output of the critical residual + BatchNorm layer combinations within the ResNet-50 and RegNetX4.0 backbones of the Faster-RCNN model.
-   **MLPArchitecture**: The auxiliary MLP is constructed as a 3-layer fully connected MLP with a single output neuron fed into a Sigmoid activation with a dropout connection before the final layer. The size for each fully connected layer is progressively halved with each consecutive layer.
-   **Transform Implementation**: During training, adversarial perturbations are applied to ID samples to create surrogate OOD data, which is used to train the MLP to detect OOD samples.

SAFE provides strong performance across almost all benchmarks and metrics, achieving the highest performance across 7 out of 8 of the benchmark permutations. Notably, they observe substantial reductions in FPR95, particularly when OpenImages is the OOD set, with a greater than 30% reduction for both backbones under the PASCAL-VOC setting.

_Note: I was going to use the code in this repository to load their retrained model, make inference on the video and include it in the AV pipeline. This repository was very promising to me since they provided pretrained SAFE models and minimalist datasets for those just wanting to use the model right away without training the whole thing again. They also had a `inference` folder which contains useful .py that I can use to make another .py file or modify those file to use the model for my data. The biggest problem that I encountered during trying to load and run the model was the environment set up. The `environment.yaml` was EXTREMELY messy and my conda could not create an environment using that file. Since I don't have enough time to dig deeper into their code to see how they build the backbone architecture and go through the yaml file to install the packages myself, I have to move on to find another paper._

----------

### 📄 2. [Unknown-Aware Object Detection: Learning What You Don't Know from Videos in the Wild](https://arxiv.org/pdf/2203.03800.pdf)  [[Code]](https://github.com/deeplearning-wisc/stud)

The authors propose a method called **STUD** (Spatial-Temporal Unknown Distillation) to improve unknown-aware object detection by learning from both the known and unknown objects in videos. The framework allows detecting unknown objects from videos without manually labeling them, which is a costly task.
-   **Videos as a Source of Unknowns**: Videos naturally contain both known objects (cars, pedestrians) and unknown ones (billboards, streetlights, etc.), making them good for capturing diverse unknowns in different frames.
-   **Distilling Unknown Objects**:
    -   **Spatial Distillation**: Unknown objects are distilled by identifying and combining them from nearby frames of a video. The method checks for objects that don’t match the known ones, assigns them a score based on their dissimilarity, and aggregates them to capture a diverse set of unknown objects.
    -   **Temporal Aggregation**: The model combines information from multiple video frames to find more diverse unknown objects over time.

To help the model learn to detect these unknown objects, the authors use an energy-based loss function. The model is trained to produce lower uncertainty scores for known objects and higher uncertainty for unknowns. This regularization pushes the decision boundaries between known and unknown objects.

They compare STUD with competitive OOD detection methods in literature, where STUD significantly outperforms baselines on both datasets. They also compare with GAN-based approach for synthe- sizing outliers in the pixel space, where STUD effectively improves the OOD detection performance (FPR95) by 15.77% on BDD100K (COCO as OOD) and 17.66% on Youtube-VIS (nulmages as OOD).

_Note: This is the second repository that I intended to use in the project pipeline. The environment for this repository was very easy to set up. They also provided different versions of the pretrained model. In the `tools` folder, there are a .py file called `train_net.py` that can be useful to make my own inference file using that file. The problem of this repository is, when I try to modify that file to ensure that I can run the `train_net.py` file in the terminal by adding a `--dummy` arg that just print out 'I can acess the file' on the screen (ie. not training or validating anything), it shows an error that said that I need some kind of .json file that needed to be generated from the dataset. Since the dataset for this project is too big for my computer to handle, I tried to look through the code base to see how can I bypass that error instead. But the project's structure and code base was too complex for me to look through everything in such a short time, I decide to look for another paper._

----------

### 📄 3. [Balanced Energy Regularization Loss for Out-of-Distribution Detection](https://openaccess.thecvf.com/content/CVPR2023/papers/Choi_Balanced_Energy_Regularization_Loss_for_Out-of-Distribution_Detection_CVPR_2023_paper.pdf)  [[Code]](https://github.com/hyunjunChhoi/Balanced_Energy)

This paper proposes a novel loss function aimed at improving out-of-distribution (OOD) detection performance by addressing class imbalance in auxiliary OOD data​. They introduced **Balanced Energy Regularization Loss** to apply higher regularization to majority classes than minority classes in auxiliary data. In other words, auxiliary samples of majority classes receive a larger energy constraint than samples of minority classes. They also introduce the term **Z**, which indicates whether a sample belongs to the majority or minority of a class. **Z** is the weighted sum of the softmax output of the classification model for a sample (i.e., the posterior prob ability of a class for a given sample), where the weight is the prior probability for the class. 

This paper have 2 pretrained model: image classification and image segmentation, but for this review, only the segmentation model will be looked at. This is how they implement the model:

- They use the semantic segmentation model of Deeplabv3+ like architecture with a WideResNet38 and employ a similar method as PEBAL. 
- They load the semantic segmentation pre trained model by NVIDIA on the Cityscapes dataset. As in PEBAL, they build the auxiliary data by cutting and pasting the mask from the COCO data.
- The model is fine-tuned using their **Balanced Energy Regularization Loss**, which adjusts the regularization applied to majority and minority classes based on the prior probabilities

This approach extends previous work on energy-based OOD detection (e.g., **EnergyOE**) by incorporating the class-wise prior distribution of OOD data into the energy-based loss. They outperforms PEBAL and achieves SOTA in a methodology that utilize OOD data and require no extra network. They present the average of all six random runs (OE, EnergyOE, Ours).15 out of 18 show better performance than baseline in AUROC, AP, and FPR.

 _Note: This is the third repository I intended to use for my project pipeline. They did upload their pretrained models and actually write out where and how to put the pretrained models in. Inside the repo, there is also a '.py' file called 'valid' that could be useful to make inference on your own. I'm not sure if you can run inference without the need of the actual dataset or not, but the problem's already arised at the moment I try to set up the environment._

----------

### 📚 Additional Papers for Future Development

Here are some of other papers I think maybe useful in future developments:

[ATTA: Anomaly-aware Test-Time Adaptation for Out-of-distribution Detection in Segmentation](https://proceedings.neurips.cc/paper_files/paper/2023/file/8dcc306a2522c60a78f047ab8739e631-Paper-Conference.pdf)  [[Code]](https://github.com/gaozhitong/ATTA)
[Residual Pattern Learning for Pixel-wise Out-of-Distribution Detection in Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Residual_Pattern_Learning_for_Pixel-Wise_Out-of-Distribution_Detection_in_Semantic_Segmentation_ICCV_2023_paper.pdf)  [[Code]](https://github.com/yyliu01/RPL)
[Scaling Out-of-Distribution Detection for Real-World Settings](https://arxiv.org/pdf/1911.11132.pdf)  [[Code]](https://github.com/hendrycks/anomaly-seg)
[VOS: Learning What You Don't Know by Virtual Outlier Synthesis](https://arxiv.org/pdf/2202.01197.pdf)  [[Code]](https://github.com/deeplearning-wisc/vos)

----------

### 🚗Part 3: A Study of Distance Measurement System

[Distance measurement system for autonomous vehicles using stereo camera](https://pdf.sciencedirectassets.com/320039/1-s2.0-S2590005619X00041/1-s2.0-S2590005620300011/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAcaCXVzLWVhc3QtMSJIMEYCIQDE87v%2BWEMkluiwt%2FdmxbeDslPQuzHIdgspUNd9w1sUFQIhALYCfu%2BpSVAJ0rLXKhYHrNPXwN%2F9cuC%2BiOceq4mxkKhJKrMFCHAQBRoMMDU5MDAzNTQ2ODY1Igw%2FX1DFuecDrJ9WTpoqkAVuUEyGC64XSd1YffWC0nwtj3wKOdljEj1i87NFkd%2BqlqUwevV4QmnFdGgjR9TmUkqdwBxrQ%2FO7bLpcjoNqPsvrWtlryMKxNItF0Eq4SNil1o0%2BSkRMnghAKXVeAxAmJxdcbtRAIGT4MJ3TezP0aYoFaZqh0XP2glM2S8CIf%2BjrmQkWsea23ksot6ownGGj31lSRMZOnXtD3zn3%2FMVBvD%2BQW4Ix4ngBfqsAiN2xYuFdYxgP9i9YtTnFz8IEUKjOd6bdkGom0SexXLH%2Froa8YJ1rotQCdPA0V6ISVV7f3Us6Ro02ZR3VQnxSuZ1L%2BGWkoOl%2FoiJg%2BybFn6l4zSHuVCMTN313DO%2BOyAvTHCGAMw5gxiRDVdoJtkglVSKpsnnywJNgJ7LrzmeKO%2FDgquT9lPZU6NudjO%2FHsgbgfJXhTAEJVJjxRhWUj23X5gk7stMSLCzNor0GDzUpBx8dQdq4ID94he6C2fEFB1gMIPgdXtU5jH4q9xzv8hw4Qv%2BzKTeI%2BZnqoiKMl21Qsk72zpoGK%2F%2BWtxAECsKR1bmFlORtuS0SeSHs7T35wdPmBR0JsMYD28orEg%2B9HOtrwrWHpOnslNV2aYw4hVcLI6jSy3hyEZ%2Fiy%2B6Nj1%2BddTvolWnFhkEGihFfJFTDTSxxXNwztz4lBZV18xak%2FEH10FkAv6%2BZwQK7iBh3N8X%2FL3li6gqyjWX7RXN6S6atp%2F8QN%2Bra5yc9wZZe2jJyhC9JWFGYY1S2X1dyHMC4z8LPRZz3ubH%2B%2BrkuLr3JunEPGN5qvksqzeRa0Qah95SXjsgoM0SYkMFzik6qHRxtanCjRBt1Wl9xgxbMU4EZQiB2CZasbF10AcNa1PP%2FabDIw5jeQpsF4zJ5%2BZyq7TCWw9K4BjqwAcCxRzcXxlv6QttzaV4eWK6496ly4NtZJx98YH2EVTy8WBXueZdOazm7V1msPdc2kfF668qv4kKGc3EaKfKHDQh1myAKMJHGPpPweQ6z2fg3JiV8mMEuwQ38qCU9ExxYCbm9O0HUnfpDqJeyueWFTjUhJ5Cb3PbOU%2FCYvbHwzWNEd2qTfafmGmlU9teKO%2BH8o%2Br2D4E0d3w4gwYvAuBC8ehuD8YH8QITlfCMPOMxlCLT&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241020T073014Z&X-Amz-SignedHeaders=host&X-Amz-Expires=299&X-Amz-Credential=ASIAQ3PHCVTYQN5S7Z66%2F20241020%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=4a541c43c61ce9485aaef475772bf1c9ef8aa57684f780465a3835a620e7579d&hash=36ff6d4acf00c0d50542f48034d02c733bac7a16d37fd3b61d8585c12aaa5657&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2590005620300011&tid=spdf-6fab025a-6cba-4f87-ac3f-761a2f66c8fb&sid=ae9a04fd893e034f8c3b94d54b28cbb3e2a8gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0c0b5f0b545750015c5554&rr=8d574d255a496bf3&cc=vn)

The paper focuses on measuring the distance between vehicles, a critical aspect for autonomous driving and safety systems. The authors propose a stereo vision-based system that uses two cameras to estimate distances by computing the disparity between the images captured by both cameras. This is how the methods work:
-   **Stereo Vision System**: The system uses two cameras mounted horizontally, similar to how human eyes perceive depth, to calculate the disparity and measure distance. This method bypasses more complex distance measurement systems like LiDAR or ultrasonic sensors by using image processing techniques.
-   **Object Detection**: The object detection method applies a cross-correlation technique on one camera’s image to detect vehicles, then matches them with the corresponding object in the second camera using stereo matching. The goal is to minimize computational load by only detecting objects in one image and matching in the other.
-   **Distance Calculation**: The system calculates the distance using simple trigonometric functions based on the cameras’ relative positions and the angular displacement of the detected object in both camera views.

The proposed system was tested in real-world scenarios, such as parking lots and roads. The experiments showed that the system provided accurate measurements of distances between the vehicle and other objects with low error margins. Additionally, the system processed up to 23 frames per second, which is fast enough for real-time applications.

_Note:_
_- For this approach, I will extract the bounding boxes and labels of the first 2 tasks. Then I will use stereo vision system to calculate each object's distance if the distance is lower than a threshold, the corresponding response will be return. (ie. it will need to stop or naviagte to the opposite direction the object is in)_
_- The actually first solution that I came up with for this part was to use a depth estimation model. Even though they are mainly used in computer graphic, I thought I can use it to extract depth map of the frame. In the first two task I would extract the bounding boxes of the known objects and unknown objects, and use those bounding boxes to put on the depth map and if the boxes fell in the areas where the values in the depth map is low, the object's indeed close to the AV, hence will need to stop or naviagte to the opposite direction the object is in. The algorithm for making decision will also check if there is any traffic light on the road, hence will also take action based on the label of the traffic light. Because of the time constraint, and the fact that the model for task 2 was not implemented yet, I can only came up with a pseudo code for demonstration. The pseudo code can be found in **'pseudo.txt'**._

____
## 🛠️ Part 4: Proposed pipeline for AV

During the span of 3 days, even though I didn't manage to finish the code base of the project, I did think about how should the pipeline be built:

 - Acquire input data from video feed from cameras mounted on the vehicle. Then convert video to frames, and perform any necessary data preprocessing.
 - Use fine-tuned DETR to process frames to detect known objects and get bounding boxes and class labels.
 - At the same time use a pre-trained OOD detection model and analyze the frame data to identify objects that are not part of the known categories.
_Note: The outputs of both above model should also be checked if they are overlapping each other._
 - Pass the outputs of both models through a stereo vision system to calculate distance or combine with outputs of depth estimation model to find objects with distance below a threshold.
 - Use the calculated object distances or close object to the camera to an algorithm to stop or steer left or right based on the placement of the object in the frame.

___
## 🔮 Part 5 : Future Works

- Fine-tune the model with larger dataset, divide the data and fine-tune session further to meet the computational resourse.
- Use Optuna to perform hyperparameter tuning, use frameworks like ONNX to reduce the model size while maintaining accuracy to run on small edge device.
- Utilize the pretrained OOD model and fine-tune if needed.
- Build a complete collision avoindance system and integrate everything into one pipeline.
- Use simulation software (like CARLA, AirSim, or Gazebo) to create diverse driving scenarios for testing.
