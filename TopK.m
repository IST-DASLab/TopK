function output = TopK(gradient,K)
[sorted_grad, I] = sort(abs(gradient),'descend'); 
output = zeros(length(gradient),1); 
for j = 1:K 
  output( I(j) ) = gradient( I(j) ); 
end
end