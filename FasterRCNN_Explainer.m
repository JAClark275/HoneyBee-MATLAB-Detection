%%
% Ver 0.0.3
% Date 23/8/2022
% 
% Jacob Clark K2141288
%% Reset Matlab enviornment
close all
clear all
clc
%% Load Network(s)

%Load the trained networks
load 'Trained_Networks\FasterRCNN_Googlenet_3Label_4.mat';
load 'Trained_Networks\FasterRCNN_ResNet50_3Label_6.mat';
load 'Trained_Networks\FasterRCNN_vgg16_3Label_6.mat';
load 'Trained_Networks\FasterRCNN_Alexnet_3Label_4.mat';

%Load the trained networks cnn subnet created manually using the network
%designer tool
load 'Trained_Networks\aCNN_subnet';
load 'Trained_Networks\gCNN_subnet';
load 'Trained_Networks\vCNN_subnet';
load 'Trained_Networks\rCNN_subnet';
%% Extract the FasterRCNN Detector
gdetector = Googlenet_3Label.outputs{1,1};
vdetector = Vgg16_3Label.outputs{1,1};
rdetector = ResNet50_3Label.outputs{1,1};
adetector = Alexnet_3Label.outputs{1,1};


%% Catagory and Feature layer setup
%Initialize the catagories of each network
category = {'Bee','Bee_Fanning','Bee_w_Pollen', 'Background'};

%Initialize the featurelayers to visualize for alexnet
afeaturelayer = {'relu1','relu2','relu3','relu4','relu5'};

%Initialize the featurelayers to visualize for googlenet split at inception
%jcution
gfeaturelayer = {'conv1-7x7_s2','conv2-3x3','inception_3a-output','inception_3b-output','inception_4a-output' 'inception_4b-output', 'inception_4c-output', 'inception_4d-output'};

%Initialize the featurelayers to visualize for vgg16
vfeaturelayer = {'relu1_2','relu2_2','relu3_3','relu4_3','relu5_3'};

%Initialize the featurelayers to visualize for Resnet it is splite because
%the amount of layer in resnet. Layers are after residual feed forward
rfeaturelayer1 = {'activation_1_relu','activation_4_relu','activation_7_relu','activation_10_relu','activation_13_relu','activation_16_relu'};
rfeaturelayer2 = {'activation_19_relu','activation_22_relu','activation_25_relu','activation_28_relu','activation_31_relu','activation_34_relu'};
rfeaturelayer3 = {'activation_37_relu','activation_40_relu','activation_43_relu','activation_46_relu','activation_49_relu'};
%% Load test images
img1 = imread('C:\Users\k2141288\OneDrive - Kingston University\ME7761_Bees\MATLAB\Aug_Multi_bee_data\Validation\017062761146.png');
img2 = imread('C:\Users\k2141288\OneDrive - Kingston University\ME7761_Bees\MATLAB\Aug_Multi_bee_data\Validation\AugImg46.jpg');

%% Call visualiztion function for each network
Visualize(img1, aCNN_subnet, adetector, "prob", category, afeaturelayer)
Visualize(img2, aCNN_subnet, adetector, "prob", category, afeaturelayer)
Visualize(img1, gCNN_subnet, gdetector, "rcnnSoftmax", category, gfeaturelayer)
Visualize(img2, gCNN_subnet, gdetector, "rcnnSoftmax", category, gfeaturelayer)
Visualize(img1, vCNN_subnet, vdetector, "prob", category, vfeaturelayer)
Visualize(img2, vCNN_subnet, vdetector, "prob", category, vfeaturelayer)
Visualize(img1, rCNN_subnet, rdetector, "rcnnSoftmax", category, rfeaturelayer1)
Visualize(img2, rCNN_subnet, rdetector, "rcnnSoftmax", category, rfeaturelayer1)
Visualize(img1, rCNN_subnet, rdetector, "rcnnSoftmax", category, rfeaturelayer2)
Visualize(img2, rCNN_subnet, rdetector, "rcnnSoftmax", category, rfeaturelayer2)
Visualize(img1, rCNN_subnet, rdetector, "rcnnSoftmax", category, rfeaturelayer3)
Visualize(img2, rCNN_subnet, rdetector, "rcnnSoftmax", category, rfeaturelayer3)

%% Deep Dream
ShowDeepDream (aCNN_subnet,'rcnnFC',1:4)
ShowDeepDream (gCNN_subnet,'rcnnFC',1:4)
ShowDeepDream (vCNN_subnet,'rcnnFC',1:4)
gpuDevice(1)%reset necesary to contibue running program
ShowDeepDream (rCNN_subnet,'rcnnFC',1:4)

%%
function ShowDeepDream (CNN_subnet,layer,channels)
    %Create an array of dreep drem images I
    I = deepDreamImage(CNN_subnet,layer,channels,...
        'PyramidLevels',2, ...
        'NumIterations',1000,...
        'Verbose',0);
    figure
    %plot for each catagory the array
    for i = 1:4
        subplot(4,1,i)
        imshow(I(:,:,:,i))
    end
end
%% Visualiztion Function
function Visualize(imgo, CNN, detector, rlayer, category, featurelayer)

    %resize the image
    img= imresize(imgo,[288 512]);
    
    %Apply the detector to thimage
    [bboxes, scores, labels]= detect(detector, img);
    
    %Resize the boxes to fit the orgional image
    bboxes2 = bboxresize(bboxes, 5);
    annotation = sprintf('%s: (Confidence = %f)', labels, scores);
    
    %Create an image with the inserted bounding boxes
    detectedImg = insertObjectAnnotation(imgo, 'rectangle', bboxes2, labels);
    figure
    imshow(detectedImg);

    %Apply the Grad CAM
    figure
    GradCamXLayer(category, featurelayer, img, CNN, rlayer)

    %Aplly the occlusion sensitivity
    figure
    OcclXCat(category, img, CNN)

end
%% Showing GRADCAM by layer
function GradCamXLayer(category, featurelayer, img, net, reductionlayer)
%Initialize a count for each catagory and feature layer
k=1;
cat_size = size(category,2);
feat_size = size(featurelayer,2);

    %Initialize the subplot
    subplot(cat_size,feat_size,k)
    for i=1:4
        for j=1:feat_size
            %Apply the GradCAM built in funtionwith inputs
            gradCAM_score_Map= gradCAM(net, img, category{i} ,...
                ReductionLayer = reductionlayer,....
                FeatureLayer = featurelayer{j});
            subplot(cat_size,feat_size,k)
            imshow(img)
            hold on
            %Overlay gradcam sore with image
            imagesc(gradCAM_score_Map,'AlphaData',0.5);
            colormap jet
            hold off
            k=k+1;
        end
    end
end
%% Showing occlusion sensitivity
function OcclXCat(category, img, net)
%Initalize count
k=1;
cat_size = size(category,2);
    for i=1:4
        %Create subplot
        subplot(1,cat_size,k)
        %Calculate occlusion sensitivity score map
        Occlusion_score_Map= occlusionSensitivity(net, img, category{i}, 'MaskSize',[5 5], 'ExecutionEnvironment', 'cpu');
        imshow(img)
        hold on
        %Overlay map ontop of image
        imagesc(Occlusion_score_Map,'AlphaData',0.5);
        colormap jet
        hold off
        k=k+1;        
    end
end