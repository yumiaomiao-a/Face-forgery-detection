clear
 
 for i=1:1000

%load features vector extracted from real images
load('TIMIT_realfeatureall.mat');

%load features vector extracted from fake images
load('TIMIT_HQ_fakefeatureall.mat');

%real label
reallabel = ones(6400,1);

%fake label
fakelabel = zeros(6400,1);

realfeature = TIMIT_realfeatureall;
fakefeature = TIMIT_HQ_fakefeatureall.mat;

%real
% 80% for training, 20% for testing
rand=randperm(6400,5120);
real_feature_train=realfeature(rand,:);
real_label_train = reallabel(rand);
realfeature(rand,:)=[];  
reallabel(rand)=[]; 
realfeature_test=realfeature;
reallabel_test=reallabel;

%fake
% 80% for training, 20% for testing
rand2=randperm(6400,5120);%ȡfakeѵ������80%
HQ_feature_train=fakefeature(rand2,:);
HQ_label_train = fakelabel(rand2);
fakefeature(rand2,:)=[];  
fakelabel(rand2)=[];  
HQfeature_test=fakefeature;
HQlabel_test=fakelabel;

%train
%label fusion
label_train = [real_label_train;HQ_label_train];
%feature fusion
feature_train = [real_feature_train;HQ_feature_train];
feature_train = feature_train(:,[1:1065]);

%test
%label fusion
label_test = [reallabel_test;HQlabel_test];
%feature fusion
feature_test = [realfeature_test;HQfeature_test];
feature_test = feature_test(:,[1:1065]);

%train RF
model=regRF_train(feature_train,label_train,500,3);%������RFģ��
%model=svmtrain(label_train,feature_train,'-s 3 -t 1 -c 2.2 -g 2.8 -p 0.1 -h 0');

%-t �˺������ͣ��˺�����������(Ĭ��2)
%0 �C���ԣ�u'v
%1 �C����ʽ��(r*u'v + coef0)^degree
%2 �C RBF������exp(-gamma|u-v|^2)
%3 �Csigmoid��tanh(r*u'v + coef0)

%RF prediction
aaaa=regRF_predict(feature_test,model);%ģ��Ԥ��ı�ǩֵ
%[aaaa,b1,c1]=svmpredict(label_test,feature_test,model);

aaaa(aaaa<0.5)=0;
aaaa(aaaa>=0.5)=1;

t = randperm(length(aaaa));
aaaa = aaaa(t);
label_test = label_test(t);


L = aaaa - label_test;
s=sum(~~L(:)); 
ans=numel(L)-s;
accurate(i) = ans./numel(L);

 end
 
accurate=sort(accurate);
accurate = accurate(500);


ROC=plot_roc(aaaa,label_test);  
disp(ROC); 

