%Tenengrad ����Sobel������ȡ���ص�ˮƽ����ʹ�ֱ������ݶ�ֵ��Tenengrad��������Ϊ���ص��ݶȵ�ƽ���ͣ���Ϊ�ݶ�����һ����ֵT���ں����������ȡ�

function score = Tenengrad(img)
%img = imread('E:\���涫��\IQA���ݿ�\tid2013\distorted_images\I01_01_1.bmp');
 img=double(img); 
 [M N]=size(img);
 I = img;
 %����sobel����gx,gy��ͼ�����������ȡͼ��ˮƽ����ʹ�ֱ������ݶ�ֵ
GX = 0;   %ͼ��ˮƽ�����ݶ�ֵ
GY = 0;   %ͼ��ֱ�����ݶ�ֵ
FI = 0;   %��������ʱ�洢ͼ��������ֵ
T  = 0;   %���õ���ֵ
 for x=2:M-1 
     for y=2:N-1 
         GX = I(x-1,y+1)+2*I(x,y+1)+I(x+1,y+1)-I(x-1,y-1)-2*I(x,y-1)-I(x+1,y-1); 
         GY = I(x+1,y-1)+2*I(x+1,y)+I(x+1,y+1)-I(x-1,y-1)-2*I(x-1,y)-I(x-1,y+1); 
         SXY= sqrt(GX*GX+GY*GY); %ĳһ����ݶ�ֵ
         %ĳһ���ص��ݶ�ֵ�����趨����ֵ���������ص㿼�ǣ���������Ӱ��
         if SXY>T 
           FI = FI + SXY*SXY;    %Tenengradֵ����
         end 
     end 
 end 
 score = FI; 
 end 

 
