% SIMPLIFIED TOP-K
% we set the mini-batch size b \in { 1,2,..., m }
% where m  is the number of data points
% K = the number of coordinates the algorithm send in each iteration
% beta and delta are tuning parameters 
function [Fcn, Coordinates_sent] = SimplifiedTopK(A,b,Iter,beta,delta,batchsize,K)
% initialize the sequences to be collected 
Fcn = zeros(Iter,1); 
Coordinates_sent  = zeros(Iter,1); 

% initialize x_0 ,v_0 , epsilon_0 
[m,n] = size(A); 
x = zeros(n,1); 
v = zeros(n,1); 
epsilon = zeros(n,1); 

Fcn(1) = LinearRegression(A,b,v); 

tstart = tic;

num_group = floor( m/batchsize ); % the number of mini-batch group 
% run the iteration 
for i = 2:Iter

% sampling the number of group with uniform distribution p_i = 1/T 
% where T is the number of groups we have 
index = floor(num_group*rand + 1);      

% take the corresponding chuck of data points to be updated
if index == num_group  
    A_chunk = A( ( batchsize*(index -1) + 1 ):(m),: );
    b_chunk = b( ( batchsize*(index -1) + 1 ):m , 1);
else
    A_chunk = A( ( batchsize*(index -1) + 1 ):(batchsize*index),: );
    b_chunk = b( ( batchsize*(index -1) + 1 ):(batchsize*index),1 );
end
gradient = SGD(A_chunk,b_chunk,v);    % compute G(v_t)
TopK_gradient = TopK(delta.*epsilon + gradient, K ); % compute TopK gradient 

epsilon = epsilon + gradient - beta.*TopK_gradient; 
v = v - beta.*TopK_gradient; 
x = x - gradient; 

time = toc(tstart); 
Fcn(i) = LinearRegression(A,b,v); 
Coordinates_sent(i) = K; 

if mod(i,1000) == 0 
fprintf('Simplied TopK SGD with batchsize = %d: Function = %d    Coordinates sent = %d     Time = %d \n',batchsize,Fcn(i),K,time);    
end


end

end


