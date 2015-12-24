function [random_theta] = randomInitialWeights( L_size, R_size)
	
	random_theta = zeros(R_size, L_size + 1);
											% dimnesion of random_theta will be [Sj+1,(Sj + 1)]
	epsillon = 0.4;
	random_theta = rand(R_size, L_size + 1)*2*epsillon - epsillon;	% rand return random matrix in [0,1] range
	% random theta will have all values between range [-epsillon, epsillon]

end