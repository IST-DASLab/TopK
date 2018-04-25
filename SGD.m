function grad = SGD(A_index,b_index,v)
grad  = A_index'*(A_index*v - b_index); 
end