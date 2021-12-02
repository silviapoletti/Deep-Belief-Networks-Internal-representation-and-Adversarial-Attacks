%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EMNIST DIGITS ADVERSARIAL %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;

%% PREPROCESSING DATASET

% load training dataset:
fname = ['emnist-digits.mat'];
load(fname);

% dataset is a structure containing train and test structures:
inputdata = dataset.train.images;      % matrix 240000x784
dataindexes = dataset.train.labels;    %  array 240000x1
testinputdata = dataset.test.images;   % matrix 40000x786
testdataindexes = dataset.test.labels; %  array 40000x1

% convert integers into double format and normalize:
inputdata = im2double(inputdata);
dataindexes = im2double(dataindexes);
testinputdata = im2double(testinputdata);
testdataindexes = im2double(testdataindexes);

%% ADVERSARIAL ATTACK

fname = 'DBN_adv_digits.mat';
load(fname)

% one-hot encoding:
tr_labels = zeros(size(inputdata,1),10);
for i = 1:size(inputdata,1)
    x = dataindexes(i);
    tr_labels(i,x+1)=1;
end  

tr_patt = inputdata;
te_labels = tr_labels;

% GRADIENT APPROXIMATION:

H1_tr = 1./(1 + exp(-tr_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(tr_patt,1),1)));

% gradient wrt first hidden layer:
minus_exp1 = exp(-tr_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(tr_patt,1),1));
g1 = minus_exp1./(1+minus_exp1).^2;  %240000x100
g1 = g1*DN.L{1}.vishid.';            %240000x100 * 100x784

% gradient wrt second hidden layer:
minus_exp2 = exp(-H1_tr*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_tr,1),1));
g2 = minus_exp2./(1+minus_exp2).^2;  %240000x500
g2 = g2*DN.L{2}.vishid.';            %240000x500 * 500x100
g2 = g2*DN.L{1}.vishid.';            %240000x100 * 100x784
s = sign(g2);

adv = zeros(size(inputdata));
for i = 1:size(inputdata,1)
    adv(i,:)= inputdata(i,:)+0.1*s(i,:);
end 

te_patt = adv;
te_patt = im2double(te_patt);

fprintf('\nGive as input to the classifier the raw images...\n');
[W0, tr_acc0, te_acc0, pred_0] = perceptron(tr_patt, tr_labels, te_patt, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc0, te_acc0);
% Training accuracy 0.866 Test accuracy 0.100 (0.1 and 0.2)

fprintf('\nGive as input to the classifier the hidden activations of layer 1..\n');
H1_tr = 1./(1 + exp(-tr_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(tr_patt,1),1)));
H1_te = 1./(1 + exp(-te_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(te_patt,1),1)));
[W1, tr_acc1, te_acc1, pred_1] = perceptron(H1_tr, tr_labels, H1_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc1, te_acc1);
% Training accuracy 0.895 Test accuracy 0.454 (0.2)
% Training accuracy 0.895 Test accuracy 0.779 (0.1)

fprintf('\nGive as input to the classifier the hidden activations of layer 2..\n');
H2_tr = 1./(1 + exp(-H1_tr*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_tr,1),1)));
H2_te = 1./(1 + exp(-H1_te*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_te,1),1)));
[W2, tr_acc2, te_acc2, pred_2] = perceptron(H2_tr, tr_labels, H2_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc2, te_acc2);
% Training accuracy 0.956 Test accuracy 0.545 (0.2)
% Training accuracy 0.956 Test accuracy 0.889 (0.1)

%% COMPARISON WITH SIMPLE GAUSSIAN NOISE

noise = zeros(size(inputdata));
for i = 1:size(inputdata,1)
    noise(i,:)= imnoise(inputdata(i,:),'gaussian',0,0.01);
end   

te_patt = noise;
te_patt = im2double(te_patt);

fprintf('\nGive as input to the classifier the raw images...\n');
[W0, tr_acc0, te_acc0, pred_0] = perceptron(tr_patt, tr_labels, te_patt, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc0, te_acc0);
% Training accuracy 0.866 Test accuracy 0.101 (0.1)
% Training accuracy 0.866 Test accuracy 0.103 (0.01)

fprintf('\nGive as input to the classifier the hidden activations of layer 1..\n');
H1_tr = 1./(1 + exp(-tr_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(tr_patt,1),1)));
H1_te = 1./(1 + exp(-te_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(te_patt,1),1)));
[W1, tr_acc1, te_acc1, pred_1] = perceptron(H1_tr, tr_labels, H1_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc1, te_acc1);
% Training accuracy 0.895 Test accuracy 0.563 (0.1)
% Training accuracy 0.895 Test accuracy 0.888 (0.01)

fprintf('\nGive as input to the classifier the hidden activations of layer 2..\n');
H2_tr = 1./(1 + exp(-H1_tr*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_tr,1),1)));
H2_te = 1./(1 + exp(-H1_te*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_te,1),1)));
[W2, tr_acc2, te_acc2, pred_2] = perceptron(H2_tr, tr_labels, H2_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc2, te_acc2);
% Training accuracy 0.956 Test accuracy 0.704 (0.1)
% Training accuracy 0.956 Test accuracy 0.949 (0.01)

%% PLOTS

% original image
figure(1)
img = inputdata(1,:); 
img = reshape(img,[28 28]);
imshow(img)
% adversarial noise
figure(2)
s_plot = s*0.5 +0.5;               % change [-1, 1] to [0,1]
img = s_plot(1,:); 
img = reshape(img,[28 28]);
imshow(img) 
% adversarial image with low epsilon
figure(3)
img = inputdata(1,:)+0.1*s(1,:);
img = reshape(img,[28 28]);
imshow(img)
% adversarial image with high epsilon
figure(4)
img = inputdata(1,:)+0.2*s(1,:);
img = reshape(img,[28 28]);
imshow(img)
% low noise image
figure(5)
img = imnoise(inputdata(1,:),'gaussian',0,0.01);
img = reshape(img,[28 28]);
imshow(img)
% high noise image
figure(6)
img = imnoise(inputdata(1,:),'gaussian',0,0.1);
img = reshape(img,[28 28]);
imshow(img)
