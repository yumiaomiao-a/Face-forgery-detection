function feat = feature_extract(im2color)
%im2color =  imread('00001.jpg');
%addpath(genpath('lbp'));
%------------------------------------------------
% Feature Computation
%-------------------------------------------------
scalenum = 5;
window = fspecial('gaussian',3,3/2); 
window = window/sum(sum(window));


%face detection
img= im2color;
FaceDetector    = buildDetector(); 
[bbox,bbimg,faces,bbfaces] = detectFaceParts(FaceDetector,img);

[a,b] = size(faces)
if a==1
  faces = cell2mat(faces);
  faces = imresize(faces,[256,256]);
else
    [c,d]= size(faces(1,:));
    [e,f]= size(faces(2,:));
    if c>d
       faces = faces(1,:);
       faces = cell2mat(faces);
       faces = imresize(faces,[256,256]);
    else
       faces = faces(2,:);
       faces = cell2mat(faces);
       faces = imresize(faces,[256,256]);
    end
end       
    
img = faces;
%figure;
%imshow(img);
imgvgg = faces;
img = rgb2gray(img);

%CSFfeature
C = [8.7, 0.6, 2000, 1.7, 0.0063, 0.0073, 0.25, 0.2];
[sci1, csfL1, csfM1, csfH1]  = getSCINxN( img,4,C(7));
CSF = csfL1.*csfM1.*csfH1;
CSF=double(CSF);
maxvalue=max(max(CSF)');
f = CSF/maxvalue; % change to range [0,1]
CSF=f;
%figure;
%imshow(CSF);
f4=mean2(CSF);
f5=std2(CSF);
f6=entropy(CSF);

%Tenengrad
f1 = Tenen(img);
f2 = Variance(img);
 

%deep feature
run  matconvnet-1.0-beta22/matconvnet-1.0-beta22/matlab/vl_setupnn
% download VGG16 model
net = dagnn.DagNN.loadobj(load('imagenet-vgg-verydeep-16.mat'));
net.conserveMemory = 0;
net.mode = 'test';

img2 = single(imgvgg); % note: 0-255 range
img2 = imresize(img2, net.meta.normalization.imageSize(1:2));
img2 = bsxfun(@minus, img2, net.meta.normalization.averageImage);
net.eval({'x0', img2});
feature_cnn1 = net.vars(38).value;
vggfeature = reshape(feature_cnn1,1,[]);
vggfeature = double(vggfeature);


%% LBP
R = 1; P = 8;
lbp_type = { 'ri' 'u2' 'riu2' };
y = 3;
mtype = lbp_type{y};
MAPPING = getmapping( P, mtype );
feat = [];
imdist=double(img);
%tic 
for itr_scale = 1:scalenum
    %im2color = imdist;
    mu            = filter2(window, imdist, 'same');
    mu_sq         = mu.*mu;
    sigma         = sqrt(abs(filter2(window, imdist.*imdist, 'same') - mu_sq));
    structdis     = (imdist-mu)./(sigma+1);
    %figure;imshow(mat2gray(structdis));
    [alpha overallstd]       = estimateggdparam(structdis(:));   % GGD
    feat  = [feat alpha overallstd];  % 
 
    %%%%%%%%%% MSCN LBP %%%%%%%%%%%%
    LBPMap = lbp_new(structdis,R,P,MAPPING,'x');
    %%%%%% gradient magnitude weighted GLBP %%%%%
    wLBPHist = [];
    
    PC = phasecong2(imdist);
    
    structdis = max(PC,structdis);
   % figure;
   %  imshow(structdis)
   
    weightmap = structdis;
    wintesity = weightmap(2:end-1, 2:end-1);
    wintensity = abs(wintesity);
    for k = 1:max(LBPMap(:))+1
        idx = find(LBPMap == k-1);
        kval = sum(wintensity(idx));
        wLBPHist = [wLBPHist kval];
    end
    wLBPHist = wLBPHist/sum(wLBPHist);
    feat = [feat wLBPHist]; 
    
    if itr_scale==1;
        feat=[feat f1 f2  f4 f5 f6 vggfeature];
    end
    
    imdist = imresize(imdist,0.5);
end
 
