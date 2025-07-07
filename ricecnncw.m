% Run the script multiple times (3 times in this case)
for loop = 1:3
    tic; % Start the timer to measure the elapsed time for each run

    close all % Close all figure windows

    % Percentage of images to use for validation and testing (0.1% in this case)
    percImgs = 0.001;

    % Load the image dataset and labels from the specified folder
    imds = imageDatastore('Rice_Image_Dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    % Split the dataset into training and validation/testing sets
    [imdsTrain, imdsRest] = splitEachLabel(imds, percImgs, 'randomized');
    [imdsValidation, imdsTest] = splitEachLabel(imdsRest, percImgs, 'randomized');

    % Get the number of training and validation images
    numTrainImages = numel(imdsTrain.Labels);
    numValidationImages = numel(imdsValidation.Labels);

    % Print the run number, number of training images, and number of validation images
    fprintf('Run %d:\n', loop);
    fprintf('Number of Training Images: %d\n', numTrainImages);
    fprintf('Number of Validation Images: %d\n', numValidationImages);

    % Load the pre-trained AlexNet network
    net = alexnet;

    % Analyze the network architecture
    analyzeNetwork(net)

    % Get the input size required by the network
    inputSize = net.Layers(1).InputSize;

    % Replace the last few layers with new ones for transfer learning
    layersTransfer = net.Layers(1:end-3);
    numClasses = numel(categories(imdsTrain.Labels));

    layers = [
        layersTransfer
        fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
        softmaxLayer
        classificationLayer
    ];

    % Define the data augmentation parameters
    pixelRange = [-30 20];
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

    % Train the network with transfer learning
    netTransfer = trainNetwork(augimdsTrain, layers, options);

    % Classify the validation set
    [YPred, scores] = classify(netTransfer, augimdsValidation);
    YValidation = imdsValidation.Labels;

    % Calculate the validation accuracy
    YValidation = imdsValidation.Labels;
    accuracy = mean(YPred == YValidation);

    % Print the validation accuracy for the current run
    fprintf('Validation Accuracy for Run %d: %.2f%%\n', loop, accuracy * 100);

    elapsed_time = toc; % Stop the timer and get the elapsed time for the current run
    fprintf('Elapsed Time for Run %d: %.2f seconds\n', loop, elapsed_time);
    fprintf('-------------------------------------------\n');
end