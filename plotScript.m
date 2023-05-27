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
[H,W] = size(f1);
snp=[0.1,0.15,0.2,0.25,0.3,0.35,0.4];
poiss_lvls=[10,15,20,25,30,35,40];
gauss_lvls=[10,15,20,25,30,35,40];

psnr_vals = zeros(3,7);
for it=1:length(poiss_lvls)
    % Generate Noisy image
    gauss = random('Normal',0,30,size(f1));
    poiss = random('Poisson',poiss_lvls(it),size(f1));
    noisy_im = f1 + gauss + poiss; 
    b=rand(size(f1));
    c=imnoise(b,'salt & pepper',0.2);
    ind0=find(c==0);
    ind1=find(c==1);
    noisy_im(ind0) = 0;
    noisy_im(ind1) = 255;
    % noisy_im = imnoise(f1,'gaussian',100);
    % noisy_im = f1 + imnoise(f1,'gaussian',0,0.03^2) + imnoise(f1,'poisson') + imnoise(f1,'salt & pepper');
    % noisy_im = double(uint8(255*noisy_im));

    % color
    % figure,imshow(mat2gray(noisy_im(:,:,1:3)));
    % grayscale
%     figure,imshow(mat2gray(noisy_im(:,:,1)));

    % color
    % im = noisy_im(:,:,1:3);
    % grayscale
    im = noisy_im(:,:,1);
    % cat(1,im2col(im(:,:,1),[8 8]),im2col(im(:,:,2),[8 8]),im2col(im(:,:,3),[8 8]));
    [m,n,p]=size(noisy_im);
    filt_im=[];
    for iter=1:size(noisy_im,3)
        filt_im(:,:,iter) = medfilt2(noisy_im(:,:,iter));
    end
    filt_im = padarray(filt_im,[4 4],0,'both');
    [m,n,p]=size(filt_im);
% figure,imshow(mat2gray(noisy_im(:,:,1:3)));
% filt_im2=filt_im;
% filt_im = filt_im2;
% for fno=1:K
    new_im=zeros(m,n);
    pca_im=zeros(m,n);
    count=zeros(m,n);
    % filt_im(:,:,2) = medfilt2(im(:,:,2));
    % filt_im(:,:,3) = medfilt2(im(:,:,3));

    % figure,imshow(mat2gray(filt_im(:,:,1:3)));
%     figure,imshow(mat2gray(filt_im(:,:,1)));
    for i=1:4:m-4
        disp(i);
        for j=1:4:n-4
            x2=min(i+7,m);
            y2=min(j+7,n);
    %         ref_patch = filt_im(i:x2,j:y2,1:3);
            ref_patch = filt_im(i:x2,j:y2,1);
            % color
    %         P=zeros(8*8*3,250);
            % grayscale
            P=zeros(8*8,5*K);
            a=ref_patch(:,:,1);
    %         b=ref_patch(:,:,2);
    %         c=ref_patch(:,:,3);
    %         vec_patch=cat(1,a(:),b(:),c(:));
            vec_patch=a(:);
            iter=1;

            % Patch matching
    %         for k=1:3:p
            for k=1:1:p
                i1=max(1,i-11);
                j1=max(1,j-11);
                i2=min(i+11,m);
                j2=min(j+11,n);
                k2=min(k+2,p);
    %             temp = filt_im(i:i2,j:j2,k:k2);
                temp = filt_im(i1:i2,j1:j2,k);
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
            sd = mean( std(P,0,1) );
            variance = sd*sd;
            % PCA based denoising
            cov=P*transpose(P);
            [eigen_vectors,eigen_values]=eig(cov);
            [diagonal,indices] = sort(diag(eigen_values),'descend');
            eigen_vectors=eigen_vectors(:,indices);
            alpha=transpose(eigen_vectors)*P;
            alpha_ref=transpose(eigen_vectors)*vec_patch(:);

            temp=alpha.^2-variance;
            temp=sum(temp,2)/size(alpha,2);
            alpha_bar=max(0,temp);
            beta= (1./(1+variance./alpha_bar)).*alpha_ref;
            denoised_vec=eigen_vectors*beta;
            denoised_patch=reshape(denoised_vec,8,8);
            pca_im(i:x2,j:y2,1) = pca_im(i:x2,j:y2,1) + denoised_patch;

            % Select elements for set Omega
            mu = mean(P,2);
            sd = std(P,0,2);
    %         ind=[];
            [omega] = find( P>mu-2*sd & P<mu+2*sd);
    %         length(ind)
            P_omega1 = zeros(size(P));
            P_omega1(omega) = P(omega);

            % apply matrix completion
            Qnew = zeros(size(P));
            Qold = Qnew;
            eps = 10^-5;
            tou = 1;
    %         tou = rand(1)+1;
            ratio = length(omega)/numel(P_omega1);
            sigmahat = mean( std(P_omega1,0,2) );
            % color
    %         newmu = (sqrt(8*8*3) + sqrt(250) )*sqrt(ratio)*sigmahat;
            % grayscale
            newmu = (sqrt(8*8) + sqrt(5*K) )*sqrt(ratio)*sigmahat;
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
                err = norm(Qnew-Qold,'fro');
    %             err = sqrt( sum(err(:).^2) );
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
%     filt_im2(:,:,fno)=mat2gray(new_im)*255;
    im = f1(:,:,1);
    cropped_im = im(1:m-8,1:n-8,1);
    
    final_im = mat2gray(new_im)*255;
    cropped_final = final_im(5:m-4,5:n-4,1);
    
    psnrDenoised = psnr(uint8(cropped_final),uint8(cropped_im));
    psnr_vals(1,it)=psnrDenoised;
    
    med_im = mat2gray(filt_im)*255;
    cropped_med = med_im(5:m-4,5:n-4,1);
    med_psnr = psnr(uint8(cropped_med),uint8(cropped_im));
    psnr_vals(2,it)=med_psnr;
    
    pca_denoised = mat2gray(pca_im)*255;
    cropped_pca = pca_denoised(5:m-4,5:n-4,1);
    pca_psnr = psnr(uint8(cropped_pca),uint8(cropped_im));
    psnr_vals(3,it)=pca_psnr;
    
end
figure,plot(poiss_lvls,psnr_vals),xlabel('Poisson noise levels'),ylabel('psnr values'),title('varying Poisson noise levels');
saveas(gcf,'all-carphone-graph-poisson.png');
    %% reconstruction

%     % final_im=new_im./count;
%     final_im=new_im;
%     % color
%     % cropped_final = final_im(1:m-4,1:n,1:3);
%     % grayscale
%     cropped_final = final_im(5:m-4,5:n-4,1);
%     cropped_final = mat2gray(cropped_final)*255;
% 
%     % color
%     % im = f1(:,:,1:3);
%     % cropped_im = im(1:m-4,1:n,1:3);
%     im = f1(:,:,1);
%     % cropped_im = im(1:m-8,1:n-8,1);
%     % cropped_im = (cropped_im);
%     % figure,imshow(mat2gray(im));
%     figure,imshow(mat2gray(cropped_final));
% 
%     % color
%     % med_filt=mat2gray(filt_im(:,:,1:3));
%     % cropped_med = med_filt(1:m-4,1:n,1:3);
%     % grayscale
%     med_filt=(filt_im(:,:,1));
%     cropped_med = med_filt(5:m-4,5:n-4,1);
%     psnrMed = psnr(uint8(mat2gray(cropped_med)*255),uint8(mat2gray(im)*255));
%     rmse=norm( im(:)-cropped_med(:) )^2
%     % med1=10*log10( ( 255^2/norm( cropped_im(:)-cropped_med(:) )^2 ) );
%     disp('Median Filtering PSNR:');
%     disp(psnrMed);
%     % disp(med1);
%     % rmse = norm(cropped_im(:) - cropped_final(:))^2;
%     % psnrDenoised = 10*log10( (255^2)/rmse );
%     psnrDenoised = psnr(uint8(mat2gray(cropped_final)*255),uint8(mat2gray(im)*255));
%     rmse=norm( im(:)-cropped_final(:) )^2
%     % med2=10*log10((255^2/norm(cropped_im(:)-cropped_final(:))^2));
%     disp('Matrix Completion PSNR:');
%     disp(psnrDenoised);
% end
%% 
% psnr_vals=zeros(K,1);
% for fno=1:K
%     final_im = filt_im2(:,:,fno);
%     cropped_final = final_im(5:m-4,5:n-4,1);
% %     cropped_final = cropped_final(5:end-4,5:end-8);
%     im = f1(:,:,fno);
%     cropped_im = im(1:m-8,1:n-8,1);
% %     figure,imshow(mat2gray(cropped_im));
% %     figure,imshow(mat2gray(cropped_final));
% %     cropped_im = cropped_im(5:end-4,5:end-8);
%     psnrDenoised = psnr(uint8(cropped_final),uint8(cropped_im));
%     psnr_vals(fno)=psnrDenoised;
% end
% [val,index]=max(psnr_vals);
% im = f1(:,:,index);
% cropped_im = im(1:m-8,1:n-8,1);
% im = noisy_im(:,:,index);
% cropped_noisy = im(1:m-8,1:n-8,1);
% final_im = filt_im2(:,:,index);
% cropped_final = final_im(5:m-4,5:n-4,1);
% figure,imshow(mat2gray(cropped_im));
% figure,imshow(mat2gray(cropped_noisy));
% figure,imshow(mat2gray(cropped_final));
% disp('PSNR:');
% disp(psnr_vals(index));
toc;