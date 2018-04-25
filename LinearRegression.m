function Fcn =  LinearRegression(A,b,v) 
Fcn = (1/2)*norm(A*v-b)^2;
end