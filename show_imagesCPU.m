
start ='image3';
name=sprintf('%s_normal',start);
path=sprintf('%s.csv',name);
M=csvread(path);
file=sprintf('%s.mat',name);
save(file,'M');
ioImg = load(file);
I  = ioImg.('M');
figure('Name',name);
imagesc(I); axis image;
colormap gray;

name=sprintf('%s_noisy',start);
path=sprintf('%s.csv',name);
M=csvread(path);
file=sprintf('%s.mat',name);
save(file,'M');
ioImg = load(file);
I  = ioImg.('M');
figure('Name',name);
imagesc(I); axis image;
colormap gray;

name=sprintf('%s_denoised',start);
path=sprintf('%s.csv',name);
M=csvread(path);
file=sprintf('%s.mat',name);
save(file,'M');
ioImg = load(file);
I  = ioImg.('M');
figure('Name',name);
imagesc(I); axis image;
colormap gray;
name=sprintf('%s_removed',start);
path=sprintf('%s.csv',name);
M=csvread(path);
file=sprintf('%s.mat',name);
save(file,'M');
ioImg = load(file);
I  = ioImg.('M');
figure('Name',name);
imagesc(I); axis image;
colormap gray;

