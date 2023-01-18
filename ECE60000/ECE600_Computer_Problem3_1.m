% Problem 3-1

for i=1:3
    for j=1:1e6
        X3(i,j)=rand;
    end
end

S3 = sum(X3,1);
Z3 = 2*(S3-3/2);
h = histogram(Z3,100,'Normalization','pdf');

p3_1 = sprintf('Probability of analytical approximation: %d',1 - normcdf((2-3/2)/(sqrt(3)*sqrt(1/12))));
disp(p3_1);

p = 0;
for i=1:100
    if h.BinEdges(i)>= (2-3/2)/(sqrt(3)*sqrt(1/12))
        p = p + h.BinCounts(i)/1e6;
    end
end
p3_2 = sprintf('Probability of numerical approximation: %d',p);
disp(p3_2);
