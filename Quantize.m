function [Q_x,k] = Quantize( x , Y )
Q_x = zeros(length(x),1); 
[B,I] = sort( abs(x) ,'descend');


k = 1; 
criterion = abs( B(k) ); 
Q_x( I(k) ) = x( I(k) ); 

while criterion < Y*norm( x )
    k = k + 1; 
    criterion = criterion + abs( B(k) ); 
    Q_x( I(k) ) = x( I(k) ); 
end

end