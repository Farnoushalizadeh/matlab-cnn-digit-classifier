% scripts/predict_single.m
% Predict the digit from a single image path

clear; clc;

% 1. Set the image path (Ensure you have an image at this location!)
% You can change '5.png' to whatever image you want to test.
imgRelativePath = fullfile('..', 'assets', 'test', '5.png'); 

% Check if the file exists
if ~isfile(imgRelativePath)
    error('Image not found: %s. Please put a digit image in assets/test/.', imgRelativePath);
end

% 2. Load the trained network
% This assumes you have already run train.m and it created the models folder
S = load(fullfile('..', 'models', 'trainedNet.mat'), 'net');
net = S.net;

% 3. Read and preprocess the image
img = imread(imgRelativePath);

% Convert to grayscale if the image is in color (RGB)
if size(img, 3) == 3
    img = rgb2gray(img);
end

% Resize to 28x28 pixels as required by the network input layer
img = imresize(img, [28 28]);

% Convert to single precision and reshape to [28 28 1]
img = im2single(img);
img = reshape(img, [28 28 1]);

% 4. Predict the digit
predictedLabel = classify(net, img);

% 5. Display the result
imshow(img, []);
title(sprintf('Predicted Digit: %s', string(predictedLabel)));
fprintf('Predicted Digit: %s\n', string(predictedLabel));