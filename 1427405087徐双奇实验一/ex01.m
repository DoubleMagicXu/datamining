
clc,clear all;
load('Dataset+for+experiment+one.mat');

plot(A(1,:),A(2,:),'*');
hold on;

x=mean(A(1,:));
y=mean(A(2,:));
m=[x;y];
plot(m(1,:),m(2,:),'or');

x=var(A(1,:));
y=var(A(2,:));
v=[x;y];
hold off;
l=zeros(length(A),length(A));
for i=1:800
    for j=1:800
        l(i,j)=((A(1,i)-A(1,j))^2+(A(2,i)-A(2,j))^2)^(0.5);
    end
end


s=zeros(length(A),length(A));
for i=1:length(A)
    for j=1:length(A)
       if(i==j)
           s(i,j)=1;
       else
           s(i,j)=(A(1,i)*A(1,j)+A(2,i)*A(2,j))/((A(1,i)^2+A(2,i)^2)^(0.5)+(A(1,j)^2+A(2,j)^2)^(0.5));
          

       end
      
    end
end

figure;

for i=1:800
    for j=1:800
        plot3(A(2,i)/A(1,i),A(2,j)/A(1,j),s(i,j),'*');
        hold on;
    end
end
hold off;

          
