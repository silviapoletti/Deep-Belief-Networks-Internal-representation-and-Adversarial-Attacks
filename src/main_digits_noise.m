%%%%%%%%%%%%%%%%%%%%%%%
% EMNIST DIGITS NOISE %
%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;

%% NETWORK SETUP

DN.layersize   = [100,200,500];        % network architecture (100,500)
DN.nlayers     = length(DN.layersize);
DN.maxepochs   = 5;                    % unsupervised learning epochs (50)
DN.batchsize   = 125;                   % mini-batch size
sparsity       = 1;                     % set to 1 to encourage sparsity on third layer
spars_factor   = 0.05;                  % how much sparsity?
epsilonw       = 0.1;                   % learning rate (weights) (0.01)
epsilonvb      = 0.1;                   % learning rate (visible biases) (0.01)
epsilonhb      = 0.1;                   % learning rate (hidden biases) (0.01)
weightcost     = 0.002;                 % decay factor (0.002)
init_momentum  = 0.5;                   % initial momentum coefficient
final_momentum = 0.9;                   % momentum coefficient

%% PREPROCESSING DATASET

% load training dataset:
fname = ['emnist-digits.mat'];
load(fname);

% dataset is a structure containing train and test structures:
inputdata = dataset.train.images;      % matrix 240000x784
dataindexes = dataset.train.labels;    % array 240000x1
testinputdata = dataset.test.images;   % matrix 40000x786
testdataindexes = dataset.test.labels; % array 40000x1

%% DATA AUGMENTATION:

inputdata = [inputdata; imnoise(inputdata(100001:150000,:),'gaussian',0,0.05)];
dataindexes = [dataindexes; dataindexes(100001:150000)];
fprintf('Noise gaussian\n');

inputdata = [inputdata; imnoise(inputdata(190001:240000,:),'salt & pepper',0.05)];
dataindexes = [dataindexes; dataindexes(190001:240000)];
fprintf('Noise local\n');

% rotation 10 degrees:
temp = zeros(25000,784); 
for i = 1:25000
    img = inputdata(i,:);             
    img = reshape(img,[28 28]);      
    img = imrotate(img,10);
    img = imresize(img,0.82);
    img = reshape(img,[1,784]);
    temp(i,:) = img;
end
inputdata = [inputdata; temp];
dataindexes = [dataindexes; dataindexes(1:25000)];
fprintf('Rotation degree 10\n');

% rotation -10 degrees:
for i = 25001:50000
    img = inputdata(i,:);                  
    img = reshape(img,[28 28]);      
    img = imrotate(img,-10);
    img = imresize(img,0.82);
    img = reshape(img,[1,784]);
    temp(i-25000,:) = img;
end
inputdata = [inputdata; temp];
dataindexes = [dataindexes; dataindexes(25001:50000)];
fprintf('Rotation degree -10\n');

% rotation 180 degrees: 6 becomes 9
idx6 = find(dataindexes(1:20135)==6);  % 200 occurrences - 1885
                                       % 2000            - 20135
data6 = inputdata(idx6,:);
temp6 = zeros(length(idx6),784);
for i = 1:length(idx6)
    img = data6(i,:);                  
    img = reshape(img,[28 28]);       
    img = imrotate(img,180);
    img = reshape(img,[1,784]);
    temp6(i,:)=img;
end
inputdata = [inputdata; temp6];
dataindexes = [dataindexes; 9*ones(length(idx6),1)];
fprintf('Rotation degree 180 of 6\n');

% rotation 180 degrees: 9 becomes 6
idx9 = find(dataindexes(1:20160)==9);  % 200 occurrences - 2160
                                       % 2000            - 20160
data9 = inputdata(idx9,:);
temp9 = zeros(length(idx9),784);
for i = 1:length(idx9)
    img = data9(i,:);                
    img = reshape(img,[28 28]);       
    img = imrotate(img,180);
    img = reshape(img,[1,784]);
    temp9(i,:)=img;
end
inputdata = [inputdata; temp9];
dataindexes = [dataindexes; 6*ones(length(idx9),1)];
fprintf('Rotation degree 180 of 9\n');

%rotation 180 degrees: 8 remains 8
idx8 = find(dataindexes(1:10210)==8);   % 150             - 1490
                                       % 500 occurrences - 5030
                                       % 1000            - 10210
                                       % 2000            - 20315
data8 = inputdata(idx8,:);
temp8 = zeros(length(idx8),784);
for i = 1:length(idx8)
    img = data8(i,:);                 
    img = reshape(img,[28 28]);      
    img = imrotate(img,180);
    img = reshape(img,[1,784]);
    temp8(i,:)=img;
end
inputdata = [inputdata; temp8];
dataindexes = [dataindexes; 8*ones(length(idx8),1)];
fprintf('Rotation degree 180 of 8\n');

% rotation 180 degrees: 0 remains 0
idx0 = find(dataindexes(1:9720)==0);   % 150             - 1380
                                       % 500 occurrences - 4940
                                       % 1000            - 9720
                                       % 2000            - 19100
data0 = inputdata(idx0,:);
temp0 = zeros(length(idx0),784);
for i = 1:length(idx0)
    img = data0(i,:);                 
    img = reshape(img,[28 28]);       
    img = imrotate(img,180);
    img = reshape(img,[1,784]);
    temp0(i,:)=img;
end
inputdata = [inputdata; temp0];
dataindexes = [dataindexes; zeros(length(idx0),1)];
fprintf('Rotation degree 180 of 0\n');

% reflection axis y: 8 remains 8
tform = maketform('affine',[1 0 0; 0 -1 0; 0 0 1]);
for i = 1:length(idx8)
    img = data8(i,:);                  
    img = reshape(img,[28 28]);      
    img = imtransform(img,tform);
    img = reshape(img,[1,784]);
    temp8(i,:)=img;
end
inputdata = [inputdata; temp8];
dataindexes = [dataindexes; 8*ones(length(idx8),1)];
fprintf('Reflection axis y of 8\n');

% reflection axis y: 0 remains 0
tform = maketform('affine',[1 0 0; 0 -1 0; 0 0 1]);
for i = 1:length(idx0)
    img = data0(i,:);                 
    img = reshape(img,[28 28]);       
    img = imtransform(img,tform);
    img = reshape(img,[1,784]);
    temp0(i,:)=img;
end
inputdata = [inputdata; temp0];
dataindexes = [dataindexes; zeros(length(idx0),1)];
fprintf('Reflection axis y of 0\n');

% convert integers into double format and normalize:
inputdata = im2double(inputdata);
dataindexes = im2double(dataindexes);
testinputdata = im2double(testinputdata);
testdataindexes = im2double(testdataindexes);

% divide the datasets in batches of size %DN.batchsize:
batchdata = reshape(inputdata.',784,DN.batchsize,[]);
batchdata = permute(batchdata,[2,1,3]);

%% GREEDY LEARNING

fprintf(1,'\nUnsupervised training of a deep belief net\n');
DN.err = zeros(DN.maxepochs, DN.nlayers, 'single');
tic();

% Greedy learning: train one layer at a time and then freeze the weights
% and move to the next layer, until we reach the last layer.
for layer = 1:DN.nlayers
    % we learn one layer at a time:
    % for the first layer, input data are raw images
    % for next layers, input data are preceding hidden activations
    fprintf(1,'Training layer %d...\n', layer);
    %fflush(stdout); % if running in Octave
    if layer == 1
        data = batchdata;
    else
        data  = batchposhidprobs;
    end
    
    % initialize weights and biases for each layer:
    numhid  = DN.layersize(layer);
    [numcases, numdims, numbatches] = size(data);
    vishid       = 0.1*randn(numdims, numhid);
    hidbiases    = zeros(1,numhid);
    visbiases    = zeros(1,numdims);
    vishidinc    = zeros(numdims, numhid);
    hidbiasinc   = zeros(1,numhid);
    visbiasinc   = zeros(1,numdims);
    batchposhidprobs = zeros(DN.batchsize, numhid, numbatches);
    
    for epoch = 1:DN.maxepochs
        errsum = 0;
        for mb = 1:numbatches
            data_mb = data(:, :, mb);
            % learn an RBM with 1-step contrastive divergence:
            rbm;
            errsum = errsum + err;   % update the error
            if epoch == DN.maxepochs
                batchposhidprobs(:, :, mb) = poshidprobs;  % poitive hidden probabilities 
            end                                            % computed during the positive phase
            if sparsity && (layer == 3)
                poshidact = sum(poshidprobs);
                Q = poshidact/DN.batchsize;
                if mean(Q) > spars_factor
                    hidbiases = hidbiases - epsilonhb*(Q-spars_factor);
                end
            end
        end
        DN.err(epoch, layer) = errsum;
    end
    % save learned weights:
    DN.L{layer}.hidbiases  = hidbiases;
    DN.L{layer}.vishid     = vishid;
    DN.L{layer}.visbiases  = visbiases;
    
end

DN.learningtime = toc();

%% SAVE FINAL NETWORK AND PARAMETERS

fprintf(1, '\nElapsed time: %d \n', DN.learningtime);
%fname = 'DBN_noise_digits.mat';
fname = 'DBN_noise_digits_aug.mat';
save(fname,'DN');
%% RECEPTIVE FIELDS

%fname = 'DBN_noise_digits.mat';
fname = 'DBN_noise_digits_aug.mat';
load(fname)

figure(1)
plot_L1(DN, 100) 
figure(2)
plot_L2(DN, 100) 
figure(3)
plot_L3(DN, 100)

%% MODEL EVALUATION

% one-hot encoding:
tr_labels = zeros(size(inputdata,1),10);
for i = 1:size(inputdata,1)
    x = dataindexes(i);
    tr_labels(i,x+1)=1;
end  

te_labels = zeros(size(testinputdata,1),10);
for i = 1:size(testinputdata,1)
    x = testdataindexes(i);
    te_labels(i,x+1)=1;
end  

tr_patt = inputdata;
te_patt = testinputdata;
te_patt(1:5000,:) = imnoise(te_patt(1:5000,:),'salt & pepper',0.08);
te_patt(5001:10000,:) = imnoise(te_patt(5001:10000,:),'gaussian',0,0.08);
for i = 10001:12000
    img = te_patt(i,:);             
    img = reshape(img,[28 28]);      
    img = imrotate(img,10);
    img = imresize(img,0.82);
    img = reshape(img,[1,784]);
    te_patt(i,:) = img;
end
for i = 12001:14000
    img = te_patt(i,:);             
    img = reshape(img,[28 28]);      
    img = imrotate(img,-10);
    img = imresize(img,0.82);
    img = reshape(img,[1,784]);
    te_patt(i,:) = img;
end

for i = 10001:12000
    img = te_patt(i,:);             
    img = reshape(img,[28 28]);      
    img = imrotate(img,20);
    img = imresize(img,0.77);
    img = reshape(img,[1,784]);
    te_patt(i,:) = img;
end
for i = 12001:14000
    img = te_patt(i,:);             
    img = reshape(img,[28 28]);      
    img = imrotate(img,-20);
    img = imresize(img,0.77);
    img = reshape(img,[1,784]);
    te_patt(i,:) = img;
end

te_patt = im2double(te_patt);

% learning analisys:
fprintf('\nGive as input to the classifier the raw images...\n');
[W0, tr_acc0, te_acc0, pred_0] = perceptron(tr_patt, tr_labels, te_patt, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc0, te_acc0);

fprintf('\nGive as input to the classifier the hidden activations of layer 1..\n');
H1_tr = 1./(1 + exp(-tr_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(tr_patt,1),1)));
H1_te = 1./(1 + exp(-te_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(te_patt,1),1)));
[W1, tr_acc1, te_acc1, pred_1] = perceptron(H1_tr, tr_labels, H1_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc1, te_acc1);

fprintf('\nGive as input to the classifier the hidden activations of layer 2..\n');
H2_tr = 1./(1 + exp(-H1_tr*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_tr,1),1)));
H2_te = 1./(1 + exp(-H1_te*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_te,1),1)));
[W2, tr_acc2, te_acc2, pred_2] = perceptron(H2_tr, tr_labels, H2_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc2, te_acc2);

fprintf('\nGive as input to the classifier the hidden activations of layer 3..\n');
H3_tr = 1./(1 + exp(-H2_tr*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_tr,1),1)));
H3_te = 1./(1 + exp(-H2_te*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_te,1),1)));
[W3, tr_acc3, te_acc3, pred_3] = perceptron(H3_tr, tr_labels, H3_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc3, te_acc3);

% w/out                  918  964 -> noisy test:            871 929
%                                 -> gauss+salt&pepper:     889 941
%                                 -> rotation 10:           902 948
%                                 -> rotation 20:           880 926
%                                 -> all                    817 863
% 5 epochs               924  963
% w/             856 879 935  
% rot+refl(500)          924
% rot+refl(1000)         899
% rot+refl(150)          929
% 15 epochs              913  920
% 20 epochs              916  909
% 10 epochs              918  943
%  5 epochs              917  961
%  5 epochs, less noise  917  961
%  5 epochs, rot 10 deg  921  961

%  ''        0.05        922  960 -> noisy test:            901 951
%                                 -> gauss+salt&pepper:     908 954
%                                 -> rotation 10:           911 956
%                                 -> rotation 20:           896 944
%  ''        + 20 deg    915  954
%  ''        0.02        925  963
