clc;clear all;
% load('瀹為獙涓夋暟鎹泦+YaleB_32x32.mat')
% for i=0:37%
%     labels(:,64*i+1)
% end
load('实验三数据集+YaleB_32x32.mat')
train_data=zeros(1024,38*30);
train_labels=zeros(1,38*30);

test_data=zeros(1024,length(A)-38*30);
test_labels=zeros(1,length(A)-38*30);
%初始化训练标签
n=1;
for i=0:37
    for j=1:30
        train_labels(1,n)=i+1;
        n=n+1;
    end
end

%初始化训练数据
% n=1;
% for i=0:37
%     for j=1:30
%         train_data(:,n)=A(:,i*64+j);
%         n=n+1;
%     end
% end
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





