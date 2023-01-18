% Problem 3-2

for i=1:30
    for j=1:1e6
        X4(i,j)=rand;
    end
end

S4 = sum(X4,1);
Z4 = (S4-30/2)/(sqrt(30)*sqrt(1/12));
h = histogram(Z4,100,'Normalization','pdf');

p3_3 = sprintf('Probability of analytical approximation: %d',1 - normcdf((20-30/2)/(sqrt(30)*sqrt(1/12))));
disp(p3_3);

p = 0;
for i=1:100
    if h.BinEdges(i)>= (20-30/2)/(sqrt(30)*sqrt(1/12))
        p = p + h.BinCounts(i)/1e6;
    end
end
p3_4 = sprintf('Probability of numerical approximation: %d',p);
disp(p3_4);