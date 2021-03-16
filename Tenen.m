%Tenengrad 采用Sobel算子提取像素点水平方向和垂直方向的梯度值，Tenengrad函数定义为像素点梯度的平方和，并为梯度设置一个阈值T调节函数的灵敏度。

function score = Tenengrad(img)
%img = imread('E:\桌面东西\IQA数据库\tid2013\distorted_images\I01_01_1.bmp');
 img=double(img); 
 [M N]=size(img);
 I = img;
 %利用sobel算子gx,gy与图像做卷积，提取图像水平方向和垂直方向的梯度值
GX = 0;   %图像水平方向梯度值
GY = 0;   %图像垂直方向梯度值
FI = 0;   %变量，暂时存储图像清晰度值
T  = 0;   %设置的阈值
 for x=2:M-1 
     for y=2:N-1 
         GX = I(x-1,y+1)+2*I(x,y+1)+I(x+1,y+1)-I(x-1,y-1)-2*I(x,y-1)-I(x+1,y-1); 
         GY = I(x+1,y-1)+2*I(x+1,y)+I(x+1,y+1)-I(x-1,y-1)-2*I(x-1,y)-I(x-1,y+1); 
         SXY= sqrt(GX*GX+GY*GY); %某一点的梯度值
         %某一像素点梯度值大于设定的阈值，将该像素点考虑，消除噪声影响
         if SXY>T 
           FI = FI + SXY*SXY;    %Tenengrad值定义
         end 
     end 
 end 
 score = FI; 
 end 

 
