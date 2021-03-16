%Variance函数表示图像灰度分布的离散程度。离焦图像灰度值变换范围小，
%离散程度低，方差小；正焦图像灰度值变换范围大，离散程度高，方差大。因此可以用其作为评价函数.
%Variance
 
function score = Variance(img)
%img = imread('E:\桌面东西\IQA数据库\tid2013\distorted_images\I01_01_1.bmp');
 img=double(img); 
 [M N]=size(img);
 I = img;  
 gama = 0;   %gama图像平均灰度值
 %求gama
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
