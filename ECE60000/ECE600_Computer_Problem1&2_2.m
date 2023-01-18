% Problem 1 & 2-1
fx = @(x) normpdf(x);

for i=1:3
    for j=1:1e6
        X1(i,j)=rand;
    end
end

S1 = sum(X1,1);
Z1 = (S1-3/2)/(sqrt(3)*sqrt(1/12));

h = histogram(Z1,100,'Normalization','pdf');
hold on
fplot(fx,[-6,6],'r','Linewidth',1);

z_1 = linspace(-3,-1);
z_2 = linspace(-1,1);
z_3 = linspace(1,3);
f_1 = 1/16*((z_1).^2+6*z_1+9);
f_2 = -1/8*z_2.^2+3/8;
f_3 = 1/16*((z_3).^2-6*z_3+9);
z = [z_1 z_2 z_3];
f = [f_1 f_2 f_3];

h1 = plot(z, f, 'g','Linewidth',1);
legend({'numerical','analytical','real'},'Location','northeast')
title('Plot of PDFs in Problem1&Problem2\_1')
xlabel('value') 
ylabel('pdf') 
hold off