%%
% Ver 0.0.3
% Date 18/8/2022
% 
% Jacob Clark K2141288
%% 
close all
%% Load the data 

%Load the test data and change the source path since labeling the data was
%done on a home computer changing the address of each file
load 'C:\Users\k2141288\OneDrive - Kingston University\ME7761_Bees\MATLAB\Aug_Multi_bee_data\Aug_GT_Validation_3Labels';
currentPathDataSource = "C:\Users\JACla\OneDrive - Kingston University\ME7761_Bees\MATLAB\Aug_Multi_bee_data\Validation";
newPathDataSource = fullfile("C:\Users\k2141288\OneDrive - Kingston University\ME7761_Bees\MATLAB\Aug_Multi_bee_data\Validation");
alternativeFilePaths = {[currentPathDataSource newPathDataSource]};
unresolvedPaths = changeFilePaths(gTruth, alternativeFilePaths);
valgtruth = gTruth;

% Apply the resizing of the data for the netowk
[valimds, valblds] = objectDetectorTrainingData(valgtruth);
valdata_temp = combine(valimds, valblds);
valdata = transform(valdata_temp, @aug_resize);

%% load up the detector
load 'Trained_Networks\FasterRCNN_Googlenet_3Label_4.mat';
load 'Trained_Networks\FasterRCNN_ResNet50_3Label_6.mat';
load 'Trained_Networks\FasterRCNN_vgg16_3Label_6.mat';
load 'Trained_Networks\FasterRCNN_Alexnet_3Label_4.mat';

%Extract the fasterRCNN from the sytcucture 
detector1 = Googlenet_3Label.outputs{1,1};
detector2 = Vgg16_3Label.outputs{1,1};
detector3 = ResNet50_3Label.outputs{1,1};
detector4 = Alexnet_3Label.outputs{1,1};

% detect the validitions
detectionResults1 = detect(detector1,valdata,'MiniBatchSize',1,'ExecutionEnvironment', 'gpu');  
detectionResults2 = detect(detector2,valdata,'MiniBatchSize',1,'ExecutionEnvironment', 'gpu'); 
detectionResults3 = detect(detector3,valdata,'MiniBatchSize',1,'ExecutionEnvironment', 'gpu'); 
detectionResults4 = detect(detector4,valdata,'MiniBatchSize',1,'ExecutionEnvironment', 'gpu');

%Initial the arrays for the mean average precision and then log mean miss
%rate
Pre_mAP1=[];
Pre_mAP2=[];
Pre_mAP3=[];
Pre_mAP4=[];
Pre_LAM1=[];
Pre_LAM2=[];
Pre_LAM3=[];
Pre_LAM4=[];

%%
%For each Intersection over union 
for IoU=.5:.05:.95
    
    % evealuate the detected data average precision and recall
    % at the IoU
    [ap1, recall1, precision1] = evaluateDetectionPrecision(detectionResults1,valdata,IoU);
    [ap2, recall2, precision2] = evaluateDetectionPrecision(detectionResults2,valdata,IoU);
    [ap3, recall3, precision3] = evaluateDetectionPrecision(detectionResults3,valdata,IoU);
    [ap4, recall4, precision4] = evaluateDetectionPrecision(detectionResults4,valdata,IoU);

    %% Plot the precision
    
    % Create a cell array of titles
    titles = {'Bee Precision', 'Bee Fanning Precision', 'Bee w Pollen Precision'};
    
    % This plot is only don whgen the IoU is .5
    if IoU ==.5
        % Start at figure 1
        figure(1)
        % For each class
        for i=1:3
            subplot(1,3,i)
            hold on
            %Plot each recal and precion for the 4 networks
            plot(recall1{i},precision1{i})
            plot(recall2{i},precision2{i})
            plot(recall3{i},precision3{i})
            plot(recall4{i},precision4{i})
            
            % Add titles and legend to the plots
            xlabel('Recall')
            ylabel('Precision')
            grid on
            title(titles{i})
            legend('Googlenet: AP = '+string(ap1(i)),...
                'Vgg16: AP = '+string(ap2(i)),...
                'ResNet50: AP = '+string(ap3(i)),...
                'AlexNet: AP = '+string(ap4(i)))
            hold off
        end
    end

    %% Miss Rate
    % evealuate the detected data average missrate and 
    % at the IoU
    [am1, fppi1, missRate1] = evaluateDetectionMissRate(detectionResults1,valdata,IoU);
    [am2, fppi2, missRate2] = evaluateDetectionMissRate(detectionResults2,valdata,IoU);
    [am3, fppi3, missRate3] = evaluateDetectionMissRate(detectionResults3,valdata,IoU);
    [am4, fppi4, missRate4] = evaluateDetectionMissRate(detectionResults4,valdata,IoU);

    % Initnalize the titles for the plors
    titles = {'Bee Miss Rate', 'Bee Fanning Miss Rate', 'Bee w Pollen Miss Rate'};
    %Only plot when th IoU is .5
    if IoU ==.5
        figure(2)
        %For Each class
        for i =1:3
            subplot(1,3,i)
            hold on
            %Plot the the FppI and missrate on a log for the 4 networks
            loglog(fppi1{i}, missRate1{i});
            loglog(fppi2{i}, missRate2{i});
            loglog(fppi3{i}, missRate3{i});
            loglog(fppi4{i}, missRate4{i});
            
            % Add titles and legend to the plots
            xlabel('FPPI')
            ylabel('MissRate')
            grid on
            title(titles{i})
            legend('Googlenet: LAM = '+string(am1(i)),...
                'Vgg16: LAM = '+string(am2(i)),...
                'ResNet50: LAM = '+string(am3(i)),...
                'AlexNet: LAM = '+string(am4(i)))
            hold off
        end
    end
    
    %For each IoU add to the averae missrate and average precision
    Pre_mAP1=[Pre_mAP1 ap1];
    Pre_mAP2=[Pre_mAP2 ap2];
    Pre_mAP3=[Pre_mAP3 ap3];
    Pre_mAP4=[Pre_mAP4 ap4];
    Pre_LAM1=[Pre_LAM1 am1];
    Pre_LAM2=[Pre_LAM2 am2];
    Pre_LAM3=[Pre_LAM3 am3];
    Pre_LAM4=[Pre_LAM4 am4];
    
end


%% Resize data function
function out= aug_resize(data)

    %Split data into it componesnt of the image boxes and labels
    Img = data{1};
    boxes = data{2};
    labels = data{3};

    %Declare the resize for the input into the networks
    rsize = [288 512];

    %Resize the image, then the boxes, no change to the labels
    augmentedImage=imresize(Img, rsize);
    augmentedBoxes=bboxresize(boxes, .2);
    augmentedLabels=labels;
    
    %Output the resized image
    out = {augmentedImage,augmentedBoxes,augmentedLabels};
end
