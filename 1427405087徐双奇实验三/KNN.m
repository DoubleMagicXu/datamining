function  output=KNN(train_data,train_labels,test_data,output_labels,c)
c_labels=ones(1,c);%这里保存 最近的c个标签，初始标签为1
d=zeros(2,size(train_data,2));%用来保存所有距离，用来排序
for i=1:size(test_data,2)
    for k=1:size(train_data,2)
        temp=0;
        for j=1:size(train_data,1)
            temp=temp+(test_data(j,i)-train_data(j,k))^2;
        end
        d(1,k)=temp;
        d(2,k)=train_labels(:,k);%标签
    end
    [~,pos]=sort(d(1,:));%位置
    for n=1:c
        c_labels(1,n)=d(2,pos(1,c));
    end
    table=tabulate(c_labels);
    [~,pos]=max(table(:,2));
    
    output_labels(:,i)=table(pos);
end
output=output_labels;
end