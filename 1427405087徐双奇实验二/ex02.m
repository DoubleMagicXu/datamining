clc,clear all;
load('实验二数据集Georgia_Tech_face_database_gray.mat');
% 对于一个图像数据集A（维度D=1024，样本数量N=750），实现：
%  （1）分别计算（第1幅vs第3幅）与（第2幅vs第20幅）图像之间的相关系数（Correlation coefficient），给出判定相关结论。
% （2）分别计算（第1幅vs第3幅）与（第2幅vs第20幅）图像之间的Co-Variance (协方差)。
% （3）对数据集A进行每维度归一化，并分别可视化出归一化前后的第1幅与第16幅图像，比较区别并给出必要的结果描述。
% （4）对数据集A中的每个样本数据进行l2归一化，并分别可视化出归一化前后的第2幅与第20幅图像，比较区别并给出必要的结果描述。
% （5）对数据集A中的每个维度或特征进行zero-mean, unit variance归一化，并分别可视化出归一化前后的第2幅与第20幅图像，比较区别并给出必要的结果描述。


%step1 Calculated correlation coefficient
sample1=A(:,1);
sample3=A(:,3);
sample2=A(:,2);
sample20=A(:,20);
cc1_3=corrcoef(sample1,sample3);
cc2_20=corrcoef(sample2,sample20);
%step 2 cov
cv1_3=cov(sample1,sample3);
cv2_20=cov(sample2,sample20);
%a=cov(sample1,sample3)./((var(sample1)*var(sample3))^0.5);
%b=cov(sample2,sample20)./((var(sample2)*var(sample20))^0.5b);

%step 3  
normalization_A=mapminmax(A, 0, 1);%normalization 0-1
%step 4 reshape
reshape1=reshape(normalization_A(:,1),32,32);
subplot(3,2,1)
imagesc(reshape1);
axis square;
title('picture01');
reshape16=reshape(normalization_A(:,16),32,32);
subplot(3,2,2)
imagesc(reshape16);
axis square;
title('picture16');

L2_A = A./repmat(sqrt(sum(A.^2,1)),size(A,1),1);%L2 normalization
reshape2=reshape(L2_A(:,2),32,32);
subplot(3,2,3)
imagesc(reshape2);
axis square;
title('picture02');
reshape20=reshape(L2_A(:,20),32,32);
subplot(3,2,4)
imagesc(reshape20);
axis square;
title('picture20');
%zero_mean
zero_mean_A=premnmx(A);
reshape2=reshape(zero_mean_A(:,2),32,32);
subplot(3,2,5)
imagesc(reshape2);
axis square;
title('zero mean,unit variance:picture02');
reshape2=reshape(zero_mean_A(:,20),32,32);
subplot(3,2,6)
imagesc(reshape20);
axis square;
title('zero mean,unit variance:picture20');