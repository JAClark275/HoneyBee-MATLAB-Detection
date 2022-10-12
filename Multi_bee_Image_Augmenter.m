%%
% Ver 0.0.2
% Date 12/7/2022
% 
% Jacob Clark K2141288
%%
%cout the data in the the multi Bee images
load Aug_Multi_bee_data\Aug_GT_training_3Labels.mat
%% Count the number of Instances of Each Label

% Count the number of labels in the ground truth
n_labels = size(gTruth.LabelDefinitions, 1);

% Initialize the a cell array of names
label_names = {};

%For each label, extract the name and place into the names 
for i=1:n_labels
    label_names{i} = char(gTruth.LabelDefinitions{i,'Name'});  
end

%Initial a matrix to count the instances of each label 
label_count = zeros(n_labels, 1);

%For each label, for the size label data or number of images, count the
%instances for which the label appears , counting the number ROI boxes
for i = 1:n_labels
    for j=1:size(gTruth.LabelData, 1)
    label_count(i, 1) = label_count(i, 1) + size(cell2mat(gTruth.LabelData{j,i}),1);
    end
end

%Plot a histogram of each instances of the data 
figure(1)
histogram('Categories',label_names, 'BinCounts',label_count);

% Ratio of instances across all images. 
label_ratio = zeros(n_labels, 1);
total_instances = sum(label_count);
for i = 1:n_labels
    label_ratio(i,1) = label_count(i,1)/total_instances;
end
%% Augment and Balance Images 
[imds,blds] = objectDetectorTrainingData(gTruth);
trainingData = combine(imds, blds);

%%
augmentedTrainingData = transform(trainingData,@Transform_erase_AddPollen);

%%
Aug_data = readall(augmentedTrainingData);
%% Show one sample augmented Image
k=1;
Aug_Imgs = Aug_data{k,1};
Aug_boxes = Aug_data{k,2};
Aug_labels =  Aug_data{k,3};
annotatedImage = insertObjectAnnotation(Aug_Imgs ,'rectangle',Aug_boxes,Aug_labels, ...
    'LineWidth',4,'FontSize',20);
Aug_table = table(Aug_boxes, Aug_labels);
figure (3)
imshow(annotatedImage)

%% Write all augmented images to a file
Aug_Size = size(Aug_data,1);
filenames = string(zeros(Aug_Size, 1));

%For each augmented image give it a file name and save it out to a folder
for k=1:Aug_Size
    filename = 'Aug_Multi_bee_data\Training\AugImg'+string(k)+'.jpg';
    filenames(k,1) = filename;
    Aug_Imgs = Aug_data{k,1};
    Aug_boxes = Aug_data{k,2};
    Aug_labels =  Aug_data{k,3};
    imwrite(Aug_Imgs,filename, 'jpg')
end

%%
function out = Transform_erase_AddPollen(data)

    Img = data{1};
    boxes = data{2};
    labels = data{3};
    
    %Apply color maske to largly blue area
    [BW, mask_RGB_Img] = createMask(Img);

    % Randomly erase between 0 and 4 vertical boxes
    r = randi([0 4], 1);
    for i=1:r
        % Generate the random window
        win = randomWindow2d(size(mask_RGB_Img),"Scale",[0.02 0.13],"DimensionRatio",[1 6;3 18]);
        hwin = diff(win.YLimits)+1;
        wwin = diff(win.XLimits)+1;
        
        %Erase the window in the image
        mask_RGB_Img = imerase(mask_RGB_Img,win,"FillValues",randi([1 255],[hwin wwin 3]));
        
        %erase box and label
        [boxes, indices] = bboxerase(boxes, [win.XLimits(1) win.YLimits(1) wwin hwin], 'EraseThreshold', .8);
        labels = labels(indices);
    end

    % Randomly erase between 0 and 4 horizontal boxes
    r = randi([0 4], 1);
    for i=1:r
        % Generate the random window
        win = randomWindow2d(size(mask_RGB_Img),"Scale",[0.02 0.13],"DimensionRatio",[6 1;18 3]);
        hwin = diff(win.YLimits)+1;
        wwin = diff(win.XLimits)+1;
        
        %Erase the window in the image
        mask_RGB_Img = imerase(mask_RGB_Img,win,"FillValues",randi([1 255],[hwin wwin 3]));
        
        %erase box and label
        [boxes, indices] = bboxerase(boxes, [win.XLimits(1) win.YLimits(1) wwin hwin], 'EraseThreshold', .8);
        labels = labels(indices);
    end
    
    
    % Define random affine transform.
    tform = randomAffine2d("XReflection",true,'Rotation',[-15 15]);
    rout = affineOutputView(size(mask_RGB_Img),tform);

    % Augment image in larger transfrom
    augmentedImage = imwarp(mask_RGB_Img,tform,"OutputView",rout, 'FillValues',randi([1 255], [1 3]));
    % Move Boxes to compensate
    [augmentedBoxes, valid] = bboxwarp(boxes,tform,rout,'OverlapThreshold',0.4);
    augmentedLabels = labels(valid);

  
    % Look for pollen bees in another data set
    pollen_imds = imageDatastore('Pollen_or_NoPollen\Pollen\Training');
    
    % Add 5 pollen carrying bees
    for t=1:5
        %Read in a random pollen bee carrying image
        img2 = readimage(pollen_imds, randi(size(pollen_imds.Files,1)));
        
        %Set up transfroms for pollen carrying bees 
        tform = randomAffine2d("XReflection",true,'Rotation',[-30 30]);
        rout2 = affineOutputView(size(img2),tform,'BoundsStyle','FollowOutput');

        %augment the bee image
        img2_Aug = imwarp(img2 ,tform,"OutputView",rout2, 'FillValues',randi([1 255], [1 3]));
    
        %Define a random window in the image fit to the the augment ed
        %image of the pollen carrying bee 
        win = randomWindow2d(size(augmentedImage),size(img2_Aug));
        hwin = diff(win.YLimits)+1;
        wwin = diff(win.XLimits)+1;
        % erase and fill with pollen image
        augmentedImage = imerase(augmentedImage,win,"FillValues",img2_Aug);
 
        %erase box and label
        [augmentedBoxes, indices] = bboxerase(augmentedBoxes, [win.XLimits(1) win.YLimits(1) wwin hwin], 'EraseThreshold', .8);
        augmentedLabels = augmentedLabels(indices);
        
        % Add boxe for pollen bee 
        augmentedBoxes = [augmentedBoxes; [win.XLimits(1) win.YLimits(1) wwin hwin]];
        augmentedLabels = [augmentedLabels; ['Bee_w_Pollen']];
    end

% Return augmented data.
out = {augmentedImage,augmentedBoxes,augmentedLabels};
end

%% Color Mask Processing 
function [BW,maskedRGBImage] = createMask(RGB)
%createMask  Threshold RGB image using auto-generated code from colorThresholder app.
%  [BW,MASKEDRGBIMAGE] = createMask(RGB) thresholds image RGB using
%  auto-generated code from the colorThresholder app. The colorspace and
%  range for each channel of the colorspace were set within the app. The
%  segmentation mask is returned in BW, and a composite of the mask and
%  original RGB images is returned in maskedRGBImage.

% Auto-generated by colorThresholder app on 15-Jul-2022
%------------------------------------------------------


% Convert RGB image to chosen color space
I = rgb2ycbcr(RGB);

% Define thresholds for channel 1 based on histogram settings
channel1Min = 0.000;
channel1Max = 185.000;

% Define thresholds for channel 2 based on histogram settings
channel2Min = 0.000;
channel2Max = 148.000;

% Define thresholds for channel 3 based on histogram settings
channel3Min = 0.000;
channel3Max = 255.000;

% Create mask based on chosen histogram thresholds
sliderBW = (I(:,:,1) >= channel1Min ) & (I(:,:,1) <= channel1Max) & ...
    (I(:,:,2) >= channel2Min ) & (I(:,:,2) <= channel2Max) & ...
    (I(:,:,3) >= channel3Min ) & (I(:,:,3) <= channel3Max);
BW = sliderBW;

% Initialize output masked image based on input image.
maskedRGBImage = RGB;

%%%%%%%%%%%%%%%%
% This is origonal code to change the BW mask to color
maskedRImage = maskedRGBImage(:, :, 1);
maskedBImage = maskedRGBImage(:, :, 2);
maskedGImage = maskedRGBImage(:, :, 3);

% Set background pixels where BW is false to random number up to 255
maskedRImage(repmat(~BW,[1 1])) = randi(255);
maskedBImage(repmat(~BW,[1 1])) = randi(255);
maskedGImage(repmat(~BW,[1 1])) = randi(255);

maskedRGBImage(:, :, 1) = maskedRImage;
maskedRGBImage(:, :, 2) = maskedBImage;
maskedRGBImage(:, :, 3) = maskedGImage;
%%%%%%%%%%%%%%
end
