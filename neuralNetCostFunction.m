function [J grad] = neuralNetCostFunction ( all_theta_rolled,...
										input_layer_size,...
										hidden_layer_size,...
										output_layer_size,...
										X,...
										y,...
										lambda )
	m = size(X,1);
	%reshaaped or unroll thetas
	theta1 = reshape( all_theta_rolled( 1 : hidden_layer_size*(input_layer_size+1) ),...
							 hidden_layer_size, (input_layer_size+1) );
	theta2 = reshape( all_theta_rolled( 1 + (hidden_layer_size*(input_layer_size+1)) : end ),...
							 output_layer_size, (hidden_layer_size+1) );

	%theta1 [hidden_layer_size, inp +1] = [hidden_layer_size , n+1]
	%theta2 [output_layer_size, hidden_layer_size +1]

	theta1_grad = zeros(size(theta1));
	theta2_grad = zeros(size(theta2));

    J = 0;

	%modify y to [m, output_layer_size] matrix from a coloumn matrix
	matY = zeros(m,output_layer_size);
	
	for iter=1:m
		matY(iter,y(iter)) = 1;
	end
	% matY is the  new matrix with one in the respective class of the correct out put and rest all zero

	% Forward Prop
	a1 = [ones(m,1), X];	%[m,n+1] after bias
							%theta1 [hidden_layer_siz, inp +1] = [hidden_layer_size , n+1]
	z2 = a1 * theta1';		%[m, hidden_layer_size]
	a2 = sigmoid(z2); 		%[m , hidden_layer_size]
	a2 = [ones(size(a2,1),1), a2];	%[m,hidden_layer_size +1] after bias
									%theta2 [out_lay_siz, hidden_layer_siz +1]
	z3 = a2 * theta2';	%	[m, output_layer_siz]
	a3 = sigmoid(z3); 	% 	[m , output_layer_size]	this is the hx for ur net
	%hx = a3;
	% ready to calculate the cost for this theta
	%Jo = -(1/m)*sum(sum(matY*log(hx) + (1-matY)*(log(1-hx)))); 
	for i = 1:m
    	for k = 1:output_layer_size
        	J =J + -matY(i, k) * log(a3(i, k)) - (1 - matY(i, k)) * log(1 - a3(i, k));
    	end
	end
	J = (1/m) * J;
	%regularization term
	theta1_fil = theta1;		% 1st col of theta is bias terms so no regularization;
	theta1_fil(1) = 0;			% so we make it 0 that it dont change the theta grad
	theta2_fil = theta2;
	theta2_fil(1) = 0;

	regu_cost = (lambda/(2*m)) * (sum(sum(theta1_fil.^2)) + sum(sum(theta2_fil.^2)));	%use theta_fil for regularize

	J = J + regu_cost;

	% now to get the feedback that is the derivative and error we are going to train the net with each example seperately
	% by going forward and backword m times and collecting the DELTA
	for iter = 1:m
		% Forward Prop for each example seperately

		a1 = [1, X(iter, :)];	%[1,input_layer_size+1] after bias
								%theta1 [hidden_layer_siz, inp +1] = [hidden_layer_size , n+1]
		z2 = a1 * theta1';		%[1, hidden_layer_size]
		a2 = sigmoid(z2); 		%[1 , hidden_layer_size]
		a2 = [1, a2];	%[1,hidden_layer_size +1] after bias
										%theta2 [out_lay_siz, hidden_layer_siz +1]
		z3 = a2 * theta2';	%	[1, output_layer_siz]
		a3 = sigmoid(z3); 	% 	[1 , output_layer_size]	this is the hx for ur net

		% Back Prop for each example seperately

		d3 = a3 - matY(iter,:);	% y should be [1, output_layer_size] as s3 ; d3 = [1, output_layer_size]
					% theta2 [output_layer_size, hidden_layer_size +1]
		d2 = (d3 * theta2) .* sigmoidGrad([1,z2]);	% (d3*theta2) = [1, hidden_layer_size +1] the first one is for bias, 
													% we add 1 to z2 to make same dimension and .* them
		d2 = d2(2:end);		% get rid of the bias term in the begining d2 = [1, hidden_layer_size]
        
		theta1_grad = theta1_grad + (d2' * a1);	%[hidden_layer_size, input_layer_size+1] = theta1 = theta1_grad
		theta2_grad = theta2_grad + (d3' * a2);	%[output_layer_size, hidden_layer_size+1] = theta2 = theta2_grad
	end

	theta1_grad = (1/m)* theta1_grad;
	theta2_grad = (1/m)* theta2_grad;


	% Regularization
	
	regu1 = (lambda/m)*theta1_fil;	% filtered theta is taken bcoz	
	regu2 = (lambda	/m)*theta2_fil;	% 1st col of theta is bias terms so no regularization;

	theta1_grad = theta1_grad + regu1;
	theta2_grad = theta2_grad + regu2;
	
	%roll gradient in grad
	grad = [theta1_grad(:);theta2_grad(:)];
end



