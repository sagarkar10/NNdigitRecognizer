function sig = sigmoid (Z)
	%Z can be of any dimension
	sig = 1.0 ./(1.0 + exp(-Z));
end