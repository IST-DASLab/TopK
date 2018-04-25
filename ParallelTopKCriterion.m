function [Fcn, Coordinates_sent] = ParallelTopKCriterion(A,b,Iter,beta,delta,num_group,Y)
% initialize the sequences to be collected 
Fcn = zeros(Iter,1); 
Coordinates_sent  = zeros(Iter,1); 

% initialize y_0 , epsilon_0^p = 0  
[m,n] = size(A); 
batchsize = floor( m/num_group ); % the number of mini-batch group 


% y_0 , epsilon_0 = 0 
y = zeros(n,1); 
epsilon = zeros(n,num_group); 
a = zeros(n,num_group); 

tstart = tic;

for k = 1:Iter
% each worker computes its own gradient and error 
for ii = 1:num_group  
    
    % find chunk to compute parts of the gradient
    if ii == num_group  
    A_chunk = A( ( batchsize*(ii -1) + 1 ):(m),: );
    b_chunk = b( ( batchsize*(ii -1) + 1 ):m , 1);
    else
    A_chunk = A( ( batchsize*(ii -1) + 1 ):(batchsize*ii),: );
    b_chunk = b( ( batchsize*(ii -1) + 1 ):(batchsize*ii),1 );
    end
    
    % update a_t^p, g_t^p, 
    local_grad = SGD(A_chunk,b_chunk,y); 
    
    a(:,ii) = delta.*epsilon(:,ii) + local_grad;   % a_t^p  
    [ a(:,ii) , K ] = Quantize( a(:,ii), Y );                % g_t^p 
    epsilon(:,ii) = epsilon(:,ii) + local_grad - beta.*a(:,ii); 
               
end

% master update the iterates 
Grad = sum(a,2); 
y = y - beta*Grad; 

time = toc(tstart); 

Fcn(k) = LinearRegression(A,b,y);
Coordinates_sent(k) = K;


if mod(k,1000) == 0 
fprintf('Parallel TopK GD criterion with num_worker = %d: Function = %d    Coordinates sent = %d     Time = %d \n',num_group,Fcn(k),K,time);    
end

end

end