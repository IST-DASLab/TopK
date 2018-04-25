function Q_x = RandomizedCoordinate( x ,K )
Q_x = zeros(length(x),1); 
num_coordinate = floor(length(x)/K); 
ii = floor(num_coordinate*rand +1 );
% find chunk of coordinates to be sent
    if ii == num_coordinate  
    Q_x(( K*(ii -1) + 1 ):length(x)) = x(( K*(ii -1) + 1 ):length(x)); 
    else
    Q_x( ( K*(ii -1) + 1 ):(K*ii) ) = x( ( K*(ii -1) + 1 ):(K*ii) );
    end
end