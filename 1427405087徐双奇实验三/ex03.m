clc;clear all;

load('实验三数据集+YaleB_32x32.mat')

%L2归一化
for i=1:size(A,2) 
    norm=0;
    for j=1:size(A,1)
        norm=norm+A(j,i)^2;
    end
    norm=norm^0.5;
    A(:,i)=A(:,i)./norm;
end






train_data=zeros(1024,38*30);
train_labels=zeros(1,38*30);
test_data=zeros(1024,length(A)-38*30);
test_labels=zeros(1,length(A)-38*30);
output_labels=zeros(1,length(A)-38*30);
%初始化训练标签
n=1;
for i=0:37
    for j=1:30
        train_labels(1,n)=i+1;
        n=n+1;
    end
end

n=1;
m=1;
for i=1:38
    for j=1:length(A)
        if(labels(1,j)==i)
            for k=0:29
                train_data(:,n)=A(:,j+k);
                n=n+1;
            end
            h=j+30;
            while(labels(1,h)==i)
                test_data(:,m)=A(:,h);
                test_labels(:,m)=i;
                h=h+1;
                m=m+1;
                if(h==length(A)+1)
                    break;
                end
            end
              break;
        end
      
    end
end







output_labels=KNN(train_data,train_labels,test_data,output_labels,5);


% 
% % knn c=1
% 
% for i=1:length(test_data)
%     d=10000000000000000000000000000000000000000;
%     for k=1:length(train_data)
%         
%         temp=0;
%         for j=1:1024
%             temp=temp+(test_data(j,i)-train_data(j,k))^2;
%         end
%         if(temp<d)
%             d=temp;
%             output_labels(:,i)=train_labels(:,k);
%         end
%     end
% end

right=0;
for i=1:length(test_labels)
    if(test_labels(:,i)==output_labels(:,i))
        right=right+1;
    end
end
knn_accuracy=right/length(output_labels)
        



%svm
% group=zeros(1,38*30);
% weight=zeros(length(test_data),38);%权重，最后比较那个元素大，就选用那个元素。
% data=train_data';%训练数据转置
% for i=0:37
%     group=zeros(1,38*30);
%     for j=i*30+1:i*30+30
%         group(:,j)=1;
%     end
%     train=svmtrain(data,group);
%     test=test_data';
%     outcome=svmclassify(train,test);
%     for n=1:length(test_data)
%         if outcome(n,1)==1
%             weight(n,i+1)=weight(1,i+1)+1;
%         else
%             %其余都加一
%             weight(n,:)=weight(n,:)+1;
%             weight(n,i+1)=weight(n,i+1)-1;
%         end
%     end
% end
% for n=1:length(test_data)
%     row=weight(n,:);
%     [temp,output_labels(1,n)]=max(row);%得到标签
% end
% %svm 精确度
% right=0;
% for i=1:length(test_labels)
%     if(test_labels(:,i)==output_labels(:,i))
%         right=right+1;
%     end
% end
% svm_accuracy=right/length(output_labels)





