%dependencies
%	sigmoid.m

function [pred] = predict(X, theta1, theta2 )

	m = size(X,1);

	%theta1 [hid_layer_siz, inp +1] = [hid_lay_size , n+1]
	%theta1 [out_lay_siz, hid_layer_siz +1]


% Forward Prop
	a1 = [ones(m,1), X];	%[m,n+1] after bias
							%theta1 [hid_layer_siz, inp +1] = [hid_lay_size , n+1]
	z2 = a1 * theta1';	%[m, hid_lay_size]
	a2 = sigmoid(z2); % [m , hid_lay_size]
	a2 = [ones(m,1), a2];	%[m,hid_lay_size +1] after bias
									%theta1 [out_lay_siz, hid_layer_siz +1]
	z3 = a2 * theta2';	%	[m, out_lay_siz]
	a3 = sigmoid(z3); % [m , out_lay_size]	this is the hx for ur net
	
	
	%convert that [m,out_lay_size] to [m,1] col vec that is chose the max of the row and select the index of it

	[max_value, max_indexes] = max(a3,[],2);
    pred = max_indexes;

end