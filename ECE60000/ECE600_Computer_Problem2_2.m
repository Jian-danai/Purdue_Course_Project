% Problem 2-2
fx = @(x) normpdf(x);

for i=1:30
    for j=1:1e6
        X2(i,j)=rand;
    end
end

S2 = sum(X2,1);
Z2 = (S2-30/2)/(sqrt(30)*sqrt(1/12));


h = histogram(Z2,100,'Normalization','pdf');
hold on
fplot(fx,[-6,6],'r','Linewidth',1);
legend({'numerical','analytical'},'Location','northeast')
title('Plot of PDFs in Problem2\_2');
xlabel('value') 
ylabel('pdf') 
hold off