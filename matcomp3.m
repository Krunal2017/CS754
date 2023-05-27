tic;
% Use K=50 for final run
K = 50;
[video,audio] = mmread('../carphone_qcif.y4m',1:K);
frames = video.frames;
f1=[];
for i=1:K
    % color
%     f1(:,:,(i-1)*3+1:(i-1)*3+3) = (frames(i).cdata);
    % grayscale
    f1(:,:,i) = rgb2gray(frames(i).cdata);
end
f1=padarray(f1,[4 4],0,'pre');
f1=padarray(f1,[4 4],0,'post');
[H,W] = size(f1);

% Generate Noisy image
gauss = random('Normal',0,30,size(f1));
poiss = random('Poisson',40,size(f1));
noisy_im = f1 + gauss + poiss; 
b=rand(size(f1));
c=imnoise(b,'salt & pepper',0.1);
ind0=find(c==0);
ind1=find(c==1);
noisy_im(ind0) = 0;
noisy_im(ind1) = 255;

% color
% figure,imshow(mat2gray(noisy_im(:,:,1:3)));
% grayscale
% figure,imshow(mat2gray(noisy_im(:,:,1)));

% color
% im = noisy_im(:,:,1:3);
% grayscale
im = noisy_im(:,:,1);
% cat(1,im2col(im(:,:,1),[8 8]),im2col(im(:,:,2),[8 8]),im2col(im(:,:,3),[8 8]));
[m,n,p]=size(noisy_im);

for iter=1:size(noisy_im,3)
    filt_im(:,:,iter) = medfilt2(noisy_im(:,:,iter));
end

new_im=zeros(size(im));
count=zeros(size(im));


% figure,imshow(mat2gray(filt_im(:,:,1:3)));
% figure,imshow(mat2gray(filt_im(:,:,1)));
for i=1:4:m-8
    disp(i);
    for j=1:4:n-4
        x2=min(i+7,m);
        y2=min(j+7,n);
%         ref_patch = filt_im(i:x2,j:y2,1:3);
        ref_patch = filt_im(i:x2,j:y2,1);
        % color
%         P=zeros(8*8*3,250);
        % grayscale
        P=zeros(8*8,250);
        a=ref_patch(:,:,1);
%         b=ref_patch(:,:,2);
%         c=ref_patch(:,:,3);
%         vec_patch=cat(1,a(:),b(:),c(:));
        vec_patch=a(:);
        iter=1;
        
        % Patch matching
%         for k=1:3:p
        for k=1:1:p
            i2=min(i+11,m);
            j2=min(j+11,n);
            k2=min(k+2,p);
%             temp = filt_im(i:i2,j:j2,k:k2);
            temp = filt_im(i:i2,j:j2,k);
            a = temp(:,:,1);
%             b = temp(:,:,2);
%             c = temp(:,:,3);
%             patch=cat(1,im2col(a,[8 8]),im2col(b,[8 8]),im2col(c,[8 8]));
            patch=im2col(a,[8 8]);
            error = sum(abs(patch - vec_patch));
            
            % Top 5 patches for frame k
            [found,ind]=mink(error,5);
            top_patch = patch(:,ind);
            P(:,iter:iter+length(ind)-1) = top_patch;
            iter=iter+length(ind);
        end
        
        % Select elements for set Omega
        mu = mean(patch,2);
        sd = std(patch,0,2);
%         ind=[];
        [omega] = find( patch>mu-2*sd & patch<mu+2*sd);
%         length(ind)
        P_omega1 = zeros(size(P));
        P_omega1(omega) = P(omega);
        
        % apply matrix completion
        Qnew = zeros(size(P));
        Qold = Qnew;
        eps = 10^-5;
        tou = 1;
%         tou = rand(1)+1;
        ratio = length(omega)/length(P_omega1);
        sigmahat = mean( std(P_omega1,0,2) );
        % color
%         newmu = (sqrt(8*8*3) + sqrt(250) )*sqrt(ratio)*sigmahat;
        % grayscale
        newmu = (sqrt(8*8) + sqrt(250) )*sqrt(ratio)*sigmahat;
        for z=1:30
            temp=zeros(size(P_omega1));
            temp2 = Qold - P_omega1;
            temp(omega) = temp2(omega);
            Rk = Qold - tou*temp2;
            [U,D,V] = svd(Rk);
            ind = find(D);
            vals = max(diag(D)-tou*newmu,0);
            D(ind) = vals;
            Qnew = U*D*V';
            err = Qnew-Qold;
            err = sqrt( sum(err(:).^2) );
            Qold = Qnew;
            if err<eps
%                 disp('true');
                break;
            end
        end
%         Qnew(omega) = P_omega1(omega);
        % restore first frame
        error = sum(abs(Qnew - vec_patch));
        [found,ind]=mink(error,1);
        vec_res = Qnew(:,ind);
%         vec_res = Qnew(:,1);
        % color
%         res = reshape(vec_res,8,8,3);
%         new_im(i:x2,j:y2,1:3) = new_im(i:x2,j:y2,1:3) + res;
%         count(i:x2,j:y2,1:3) = count(i:x2,j:y2,1:3) + 1;
        % grayscale
        res = reshape(vec_res,8,8);
%         new_im(i:x2,j:y2,1) = res;
        new_im(i:x2,j:y2,1) = new_im(i:x2,j:y2,1) + res;
        count(i:x2,j:y2,1) = count(i:x2,j:y2,1) + 1;
    end
end
%% reconstruction

% final_im=new_im./count;
final_im=new_im;
% color
% cropped_final = final_im(1:m-4,1:n,1:3);
% grayscale
cropped_final = final_im(1:m-4,1:n,1);
cropped_final = cropped_final(5:end-4,5:end-8);
% cropped_final = mat2gray(cropped_final)*255;

% color
% im = f1(:,:,1:3);
% cropped_im = im(1:m-4,1:n,1:3);
im = f1(:,:,1);
cropped_im = im(1:m-4,1:n,1);
cropped_im = cropped_im(5:end-4,5:end-8);
cropped_im = (cropped_im);
inp=noisy_im(:,:,1);
inp = inp(1:m-4,1:n,1);
inp = inp(5:end-4,5:end-8);
figure,imshow(mat2gray(inp));

% color
% med_filt=mat2gray(filt_im(:,:,1:3));
% cropped_med = med_filt(1:m-4,1:n,1:3);
% grayscale
med_filt=(filt_im(:,:,1));
cropped_med = med_filt(1:m-4,1:n,1);
cropped_med = cropped_med(5:end-4,5:end-8);
figure,imshow(mat2gray(cropped_med));
figure,imshow(mat2gray(cropped_final));
psnrMed = psnr(uint8(mat2gray(cropped_med)*255),uint8(mat2gray(cropped_im)*255));

disp('Median Filtering PSNR:');
disp(psnrMed);

psnrDenoised = psnr(uint8(mat2gray(cropped_final)*255),uint8(mat2gray(cropped_im)*255));

disp('Matrix Completion PSNR:');
disp(psnrDenoised);

toc;