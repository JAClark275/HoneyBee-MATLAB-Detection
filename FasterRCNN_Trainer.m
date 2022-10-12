%%load the ground truth labes

load 'Multi_bee_data\Multi_bee_labels.mat';
%%
%Create image data store with ground truth
[imds,blds] = objectDetectorTrainingData(gTruth);

data = combine(imds, blds);
trndata = transform(data, @aug_resize);
%%
%Load the network
%load 'Untrained_Networks\FasterRCNN_ResNet50_act40.mat';

%%
network = fasterRCNNLayers([180 320 3], 1, [23, 38; 38, 23],'alexnet') 
%%
options = trainingOptions('sgdm',...
    'MaxEpochs',1,...
    'MiniBatchSize',30,...
    'InitialLearnRate',1e-3,...
    'shuffle', 'every-epoch',...
    'ValidationData',trndata, ...
    'ExecutionEnvironment', 'cpu');

%% Train fasterRCNN
[detector, info] = trainFasterRCNNObjectDetector(trndata,network,options);

%%
plot(info)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img = imread('C:\Users\k2141288\OneDrive - Kingston University\ME7761_Bees\MATLAB\Multi_bee_data\017062764698.jpg');

[bboxes, scores, labels]= detect(detector, img);

annotation = sprintf('%s: (Confidence = %f)', labels, scores);

detectedImg = insertObjectAnnotation(img, 'rectangle', bboxes, labels);

figure
imshow(detectedImg)
%%
function out= aug_resize(data)
Img = data{1};
boxes = data{2};
labels = data{3};

rsize = [180 320];

augmentedImage=imresize(Img, rsize);
augmentedBoxes=bboxresize(boxes, .125);
augmentedLabels=labels;

out = {augmentedImage,augmentedBoxes,augmentedLabels};
end