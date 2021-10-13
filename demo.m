clc
clear all
lb = -100.*ones(1,10);
ub = 100.*ones(1,10);
maxit = 1000;
objf= @Sphere;
n = 50;
d = 10;
for i=1:n
    X(i,:)=lb+rand(1,d).*(ub-lb);
end
[BestCost,BestValue]=CCLNNA(objf,n,d,lb,ub,maxit,X);
plot(BestCost,'r','linewidth',2)
xlabel('The number of iterations','Fontname','Times New Roma','fontsize',15);
ylabel('Fitness value','Fontname','Times New Roman','fontsize',15);