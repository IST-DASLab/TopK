% We solve the linear regression problem: 
% minimize (1/2)*\| Ax - b \|^2
% with the topK SGD 


% Generate the data set where 
% A is the matrix with m by n
% Each row of A represents each data point
% b is the vector with dimension m 
% Each element of b is the class label  \in {-1,1}. 
m = 1000;   n = 100; 
A = rand(m,n); 
b = sign(randn(m,1)); 

% Normalize each data point of A by its Euclidean norm
for i = 1:m
  A(i,:) = A(i,:)/norm(A(i,:));
  if mod(i,1000) == 0  
  fprintf('Normalize the data at row %d \n', i );     
  end    
end

%%
% set up the number of iterations 
Iter = 500000;
% tune the parameters for the simplified TopK SGD
K = 50; 

% we vary the batch size 
batchsize = floor([m/5;m/3;m/2]); 
delta = 0.0001; beta = 0.00001; 
Fcn_varyBatchsize =  zeros(Iter,length(batchsize)); 
Coordinates_varyBatchsize = zeros(Iter,length(batchsize));  

for bb = 1:length(batchsize)
[Fcn_varyBatchsize(:,bb), Coordinates_varyBatchsize(:,bb)] = ...
    SimplifiedTopK(A,b,Iter,beta,delta,batchsize(bb),K);
end





%% we vary delta
batchsize = floor([m/4]); 
delta = [0.0001;0.001;0.1]; 
beta = 0.00001; 
Fcn_varydelta =  zeros(Iter,length(delta)); 
Coordinates_varydelta = zeros(Iter,length(delta));  

for bb = 1:length(delta)
[Fcn_varydelta(:,bb), Coordinates_varydelta(:,bb)] = ...
    SimplifiedTopK(A,b,Iter,beta,delta(bb),batchsize,K);
end


%% we vary beta
batchsize = floor([m/4]); 
delta = 0.0001; 
beta = [0.00001;0.0001; 0.001]; 
Fcn_varybeta =  zeros(Iter,length(beta)); 
Coordinates_varybeta = zeros(Iter,length(beta));  

for bb = 1:length(beta)
[Fcn_varybeta(:,bb), Coordinates_varybeta(:,bb)] = ...
    SimplifiedTopK(A,b,Iter,beta(bb),delta,batchsize,K);
end

%% plot the result 
% plot the result when we vary batchsize 
figure()
grid('on')
hold on
semilogy(Fcn_varyBatchsize(:,1),...
          'color','b','linestyle',':','linewidth',2); 
semilogy(Fcn_varyBatchsize(:,2),...
          'color','k','linestyle','--','linewidth',2);      
semilogy(Fcn_varyBatchsize(:,3),...
         'color','k','linestyle','-','linewidth',2); 
xlabel('iteration counts$/m$','Interpreter','Latex');
ylabel('$f(x_k)$','Interpreter','Latex')
l= legend('TopK SGD batchsize = $m/5$',...
          'TopK SGD batchsize = $m/3$',...
          'TopK SGD batchsize = $m/2$');
set(l,'Interpreter','Latex','FontSize',8);     

%%
% plot the result when we vary delta
figure()
grid('on')
hold on
semilogy(Fcn_varydelta(:,1),...
          'color','b','linestyle',':','linewidth',2); 
semilogy(Fcn_varydelta(:,2),...
          'color','k','linestyle','--','linewidth',2);      
semilogy(Fcn_varydelta(:,3),...
         'color','k','linestyle','-','linewidth',2); 
xlabel('iteration counts$/m$','Interpreter','Latex');
ylabel('$f(x_k)$','Interpreter','Latex')
l= legend('TopK SGD  $\delta = 1e-3$',...
          'TopK SGD  $\delta = 1e-2$',...
          'TopK SGD  $\delta = 1e-1$');
set(l,'Interpreter','Latex','FontSize',8);    


%% plot the result when we vary beta
figure()
grid('on')
hold on
semilogy(Fcn_varybeta(:,1),...
          'color','b','linestyle',':','linewidth',2); 
semilogy(Fcn_varybeta(:,2),...
          'color','k','linestyle','--','linewidth',2);      
semilogy(Fcn_varybeta(:,3),...
        'color','k','linestyle','-','linewidth',2); 
xlabel('iteration counts$/m$','Interpreter','Latex');
ylabel('$f(x_k)$','Interpreter','Latex')
l= legend('TopK SGD  $\beta = 1e-5$',...
          'TopK SGD  $\beta = 1e-4$',...
          'TopK SGD  $\beta = 1e-3$');
set(l,'Interpreter','Latex','FontSize',8);    
