%dependencies
%	randomInitialWeights.m
%	fmincg.m
%	neuralNetCostFunction.m
%	predict.m

%inital-set
close all;
clear;
clc

%important variables
input_layer_size = 400;	% 20x20 pixel
%number_of_layer = 3;
hidden_layer_size = 30;
output_layer_size = 10;	%1 to 10 , 10 corresponds to number 0


% skiping display part


%load feature
%X,y are in .mat file and each row has each complete example

load('ex4data1.mat');

%no of examples and features
m = size(X,1);	% #.examples
n = size(X,2);	% #.features

% load('ex4weights.mat');
% 
% % Unroll parameters 
% nn_params = [Theta1(:) ; Theta2(:)];
% 
% % Weight regularization parameter (we set this to 0 here).
% lambda = 0;
% 
% J = neuralNetCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
%                    output_layer_size, X, y, lambda);
% 
% fprintf(['Cost at parameters (loaded from ex4weights): %f '...
%          '\n(this value should be about 0.287629)\n'], J);
% 
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

init_theta1 = randomInitialWeights(input_layer_size,hidden_layer_size);		% takes dimension of theta(not exactly)
init_theta2 = randomInitialWeights(hidden_layer_size,output_layer_size);  % and returns random value range see inside function

% all thetas here
rolled_init_theta = [init_theta1(:);init_theta2(:)];	% initial unrolled parameters for fmincg, it is a long col vec

%advance optimization algorithm fmincg

options = optimset('MaxIter',60);
lambda = 1;
% [unroll_init_theta, cost] = fmincg( @(p) neuralNetCostFunction(p,...
% 														input_layer_size,...
% 														hidden_layer_size,...
% 														output_layer_size,...
% 														X,...
% 														y,...
% 														lambda)...
% 									,unroll_init_theta, options);

%above code shortend
costfunc = @(p) neuralNetCostFunction(p, input_layer_size, hidden_layer_size, output_layer_size, X, y, lambda);
%its made a reffernce to the function and with the required parameters

%thus costfunc now only depends on p ie. rolled theta 
[rolled_init_theta, cost] = fmincg( costfunc, rolled_init_theta, options);

%fmincg returns the learned theta of size(rolled_init_theta) and the valuse of costfunction at that thetas

%reshaaped or unroll thetas
Theta1 = reshape( rolled_init_theta( 1 : hidden_layer_size*(input_layer_size+1) ),...
							 hidden_layer_size, (input_layer_size+1) );
Theta2 = reshape( rolled_init_theta(1+hidden_layer_size*(input_layer_size+1) : end ),...
							 output_layer_size, (hidden_layer_size+1) );

% get predictions in a vector same dimension as y
% prediction -> return a col vector[m:1] with the prediction
predictions = predict(X, Theta1, Theta2);

acuuracy = mean(double(y==predictions))*100	%think, as only one we need.... mean = sum/number