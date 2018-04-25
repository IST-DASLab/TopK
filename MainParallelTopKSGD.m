% main Parallel TopK SGD 

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

% set up the number of iterations 
Iter = 200000;
% tune the parameters for the simplified TopK SGD
K = [25, 100]; 

%% we vary delta
num_group = 8; 
delta = 0.00001; 
beta  = 0.00001; 

RandomizedK_K = zeros(Iter,length(K)); 
TopK_K = zeros(Iter,length(K));


%Y = 10;     % select 10% of the gradient 
%Fcn_TopKCriterion_delta = zeros(Iter,length(delta)); 
%Coordinates_sent_TopKCriterion_delta = zeros(Iter,length(delta));


for s = 1:length(K)
    if K(s) ~= 100
        %This is skipped for K = 100% so we don't execute it twice
        [RandomizedK_K(:,s)] = ...
            ParallelGradientSparfication(A,b,Iter,beta,delta ,num_group, K(s)); 
    end

    [TopK_K(:,s)] =...
        ParallelTopKSGD(A,b,Iter,beta,delta,num_group,K(s)); 
end


save MainParallelTopKSGDTestK.mat

% plot the result 
% 
figure()
grid('on')
hold on
semilogy(RandomizedK_K(:,1),...
          'color','b','linestyle',':','linewidth',2); 
semilogy(TopK_K(:,1),...
          'color','r','linestyle','--','linewidth',2);      
semilogy(TopK_K(:,2),...
         'color','g','linestyle','-','linewidth',2); 
xlabel('iteration counts$/m$','Interpreter','Latex');
ylabel('$f(x_k)$','Interpreter','Latex')
l= legend('Algorithm 1: RandomK $k = 25\%$',...
          'Algorithm 1: TopK $k = 25\%$',...
          'Algorithm 1: TopK $k = 100\%$');
set(l,'Interpreter','Latex','FontSize',8);     
% % 
% % plot the result when we vary beta
% % figure()
% % grid('on')
% % hold on
% % semilogy([0:(Iter-1)]/m, Fcn_beta(:,1),...
% %           'color','b','linestyle',':','linewidth',2); 
% % semilogy([0:(Iter-1)]/m, Fcn_beta(:,2),...
% %           'color','k','linestyle','--','linewidth',2);      
% % semilogy([0:(Iter-1)]/m, Fcn_beta(:,3),...
% %          'color','k','linestyle','-','linewidth',2); 
% % xlabel('iteration counts$/m$','Interpreter','Latex');
% % ylabel('$f(x_k)$','Interpreter','Latex')
% % l= legend('Algorithm 1: $\beta = 5e-4$',...
% %           'Algorithm 1: $\beta = 1e-4$',...
% %           'Algorithm 1: $\beta = 1e-3$');
% % set(l,'Interpreter','Latex','FontSize',8);  