% Run the script multiple times (3 times in this case)
for loop = 1:3
    tic; % Start the timer

    close all
    % Unzip the dataset file (if needed)
    %unzip('rice.zip');

    % Percentage of images to use for validation and testing
    percImgs = 0.001;

    % Load the image dataset and labels from the specified folder
    imds = imageDatastore('Rice_Image_Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    % Split the dataset into training, validation, and testing sets
    [imdsTrain, imdsRest] = splitEachLabel(imds, percImgs, 'randomized');
    [imdsValidation, imdsTest] = splitEachLabel(imdsRest, percImgs, 'randomized');

    % Get the number of training and validation images
    numTrainImages = numel(imdsTrain.Labels);
    numValidationImages = numel(imdsValidation.Labels);

    fprintf('Run %d:\n', loop);
    fprintf('Number of Training Images: %d\n', numTrainImages);
    fprintf('Number of Validation Images: %d\n', numValidationImages);

    % Load the pre-trained GoogLeNet network
    net = googlenet;

    % Get the input size required by the network
    inputSize = net.Layers(1).InputSize;

    % Replace the last three layers of the network with new ones for transfer learning
    lgraph = layerGraph(net);
    lgraph = removeLayers(lgraph, {'loss3-classifier', 'prob', 'output'});

    numClasses = numel(categories(imdsTrain.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses, 'Name', 'fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classoutput')
    ];
    lgraph = addLayers(lgraph, newLayers);

    % Connect the new layers to the rest of the network
    lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'fc');

    % Define the data augmentation parameters
    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter('RandXReflection', true, 'RandXTranslation', pixelRange, 'RandYTranslation', pixelRange);

    % Create augmented image datastores for training and validation
    augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

    % Define the training options
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 10, ...
        'MaxEpochs', 6, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augimdsValidation, ...
        'ValidationFrequency', 3, ...
        'Verbose', false);

    % Train the network using the layer graph
    netTransfer = trainNetwork(augimdsTrain, lgraph, options);

    % Classify the validation set
    [YPred, scores] = classify(netTransfer, augimdsValidation);
    YValidation = imdsValidation.Labels;

    % Calculate the validation accuracy
    YValidation = imdsValidation.Labels;
    accuracy = mean(YPred == YValidation);
    fprintf('Validation Accuracy for Run %d: %.2f%%\n', loop, accuracy * 100);

    elapsed_time = toc; % Stop the timer
    fprintf('Elapsed Time for Run %d: %.2f seconds\n', loop, elapsed_time);
    fprintf('-------------------------------------------\n');
end