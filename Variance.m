%Variance������ʾͼ��Ҷȷֲ�����ɢ�̶ȡ��뽹ͼ��Ҷ�ֵ�任��ΧС��
%��ɢ�̶ȵͣ�����С������ͼ��Ҷ�ֵ�任��Χ����ɢ�̶ȸߣ��������˿���������Ϊ���ۺ���.
%Variance
 
function score = Variance(img)
%img = imread('E:\���涫��\IQA���ݿ�\tid2013\distorted_images\I01_01_1.bmp');
 img=double(img); 
 [M N]=size(img);
 I = img;  
 gama = 0;   %gamaͼ��ƽ���Ҷ�ֵ
 %��gama
 for x=1:M 
     for y=1:N 
         gama = gama + I(x,y); 
     end 
 end 
 gama = gama/(M*N); 
  
 FI=0; 
 for x=1:M 
     for y=1:N 
         FI=FI+(I(x,y)-gama)*(I(x,y)-gama); 
     end 
 end 
  score = FI;

 end 
