%%%%%%%%%%%%%%%%%
% EMNIST DIGITS %
%%%%%%%%%%%%%%%%%

clear all; close all;

%% NETWORK SETUP

DN.layersize   = [100,200,500];         % network architecture
DN.nlayers     = length(DN.layersize);
DN.maxepochs   = 30;                    % unsupervised learning epochs
DN.batchsize   = 125;                   % mini-batch size
sparsity       = 1;                     % set to 1 to encourage sparsity on third layer
spars_factor   = 0.05;                  % how much sparsity?
epsilonw       = 0.1;                   % learning rate (weights)
epsilonvb      = 0.1;                   % learning rate (visible biases)
epsilonhb      = 0.1;                   % learning rate (hidden biases)
weightcost     = 0.002;                 % decay factor
init_momentum  = 0.5;                   % initial momentum coefficient
final_momentum = 0.9;                   % momentum coefficient

%% PREPROCESSING DATASET

% load training dataset:
fname = 'emnist-digits.mat';
load(fname);

% dataset is a structure containing train and test structures:
inputdata = dataset.train.images;      % matrix 240000x784
dataindexes = dataset.train.labels;    %  array 240000x1
testinputdata = dataset.test.images;   % matrix  40000x786
testdataindexes = dataset.test.labels; %  array  40000x1

% convert integers into double format and normalize:
inputdata = im2double(inputdata);
dataindexes = im2double(dataindexes);
testinputdata = im2double(testinputdata);
testdataindexes = im2double(testdataindexes);

% some examples from the dataset:
% for i = 1:10
%     test_img = inputdata(i,:);                  % get the i-th input
%     test_img = reshape(test_img,[28 28]);       % reshape the array into a matrix
%     imshow(test_img,[]);
%     caption = sprintf('EMNIST Digits\n label %d',dataindexes(i));
%     title(caption, 'FontSize', 10);
%     drawnow;
%     pause(0.5);
% end

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

% save final network and parameters:
fprintf(1, '\nElapsed time: %d \n', DN.learningtime);
% 5.042248e+02
fname = 'DBN_emnist_digits.mat';
save(fname, 'DN');

%% RECEPTIVE FIELDS

fname = 'DBN_emnist_digits.mat';
load(fname)

figure(1)
plot_L1(DN, 50) 
figure(2)
plot_L2(DN, 50) 
figure(3)
plot_L3(DN, 50) 

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

% learning analisys:
fprintf('\nGive as input to the classifier the raw images...\n');
[W0, tr_acc0, te_acc0, pred_0] = perceptron(tr_patt, tr_labels, te_patt, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc0, te_acc0);
% Training accuracy 0.866 Test accuracy 0.864
fprintf('\nGive as input to the classifier the hidden activations of layer 1..\n');
H1_tr = 1./(1 + exp(-tr_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(tr_patt,1),1)));
H1_te = 1./(1 + exp(-te_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(te_patt,1),1)));
[W1, tr_acc1, te_acc1, pred_1] = perceptron(H1_tr, tr_labels, H1_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc1, te_acc1);
% Training accuracy 0.887 Test accuracy 0.888
fprintf('\nGive as input to the classifier the hidden activations of layer 2..\n');
H2_tr = 1./(1 + exp(-H1_tr*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_tr,1),1)));
H2_te = 1./(1 + exp(-H1_te*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_te,1),1)));
[W2, tr_acc2, te_acc2, pred_2] = perceptron(H2_tr, tr_labels, H2_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc2, te_acc2);
% Training accuracy 0.931 Test accuracy 0.931
fprintf('\nGive as input to the classifier the hidden activations of layer 3..\n');
H3_tr = 1./(1 + exp(-H2_tr*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_tr,1),1)));
H3_te = 1./(1 + exp(-H2_te*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_te,1),1)));
[W3, tr_acc3, te_acc3, pred_3] = perceptron(H3_tr, tr_labels, H3_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc3, te_acc3);
% Training accuracy 0.967 Test accuracy 0.967
figure(4)
bar([te_acc0 te_acc1 te_acc2 te_acc3])
ylim([0.8 1]);
ylabel('Test accuracy')
xticklabels({'Pixels', 'H1', 'H2', 'H3'})
figure(5)
bar([tr_acc0 tr_acc1 tr_acc2 tr_acc3])
ylim([0.8 1]);
ylabel('Train accuracy')
xticklabels({'Pixels', 'H1', 'H2', 'H3'})

%% FEATURE VISUALIZATION ON PYTHON

H.first = H1_tr(1:5000,:);
H.second = H2_tr(1:5000,:);
H.third = H3_tr(1:5000,:);
fname = 'features.mat';
save(fname, 'H');
load(fname)

%% CONFUSION MATRIX

true_labels = dataset.test.labels;

% layer H1:
figure(6)
pred_labels = pred_1 - ones(size(pred_1));
cm = confusionchart(true_labels, pred_labels);
cm.Title = 'Confusion matrix layer H1';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% layer H2:
figure(7)
pred_labels = pred_2 - ones(size(pred_2));
cm = confusionchart(true_labels, pred_labels);
cm.Title = 'Confusion matrix layer H2';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% layer H3:
figure(8)
pred_labels = pred_3 - ones(size(pred_3));
cm = confusionchart(true_labels, pred_labels);
cm.Title = 'Confusion matrix layer H3';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% NOISE IN TEST IMAGES

% Gaussian noise injection to test dataset:
fprintf('\n Performance on noisy Test dataset:\n');
tr_patt = inputdata;
%te_patt = imnoise(testinputdata,'gaussian',0,0.05);
te_patt = imnoise(testinputdata,'gaussian',0.1,0.1);

% some examples from the noisy test set:
% for i = 1:10
%     test_img = te_patt(i,:);                    % get the i-th input
%     test_img = reshape(test_img,[28 28]);       % reshape the array into a matrix
%     imshow(test_img);
%     testdataindexes(i)                          % i-th label
%     pause(0.1);
% end

% learning analisys:
fprintf('\nGive as input to the classifier the raw images...\n');
[W0, tr_acc0, te_acc0, pred_0] = perceptron(tr_patt, tr_labels, te_patt, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc0, te_acc0);
% Training accuracy 0.866 Test accuracy 0.102
fprintf('\nGive as input to the classifier the hidden activations of layer 1..\n');
H1_tr = 1./(1 + exp(-tr_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(tr_patt,1),1)));
H1_te = 1./(1 + exp(-te_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(te_patt,1),1)));
[W1, tr_acc1, te_acc1, pred_1] = perceptron(H1_tr, tr_labels, H1_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc1, te_acc1);
%Training accuracy 0.887 Test accuracy 0.590
fprintf('\nGive as input to the classifier the hidden activations of layer 2..\n');
H2_tr = 1./(1 + exp(-H1_tr*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_tr,1),1)));
H2_te = 1./(1 + exp(-H1_te*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_te,1),1)));
[W2, tr_acc2, te_acc2, pred_2] = perceptron(H2_tr, tr_labels, H2_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc2, te_acc2);
% Training accuracy 0.931 Test accuracy 0.634
fprintf('\nGive as input to the classifier the hidden activations of layer 3..\n');
H3_tr = 1./(1 + exp(-H2_tr*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_tr,1),1)));
H3_te = 1./(1 + exp(-H2_te*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_te,1),1)));
[W3, tr_acc3, te_acc3, pred_3] = perceptron(H3_tr, tr_labels, H3_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc3, te_acc3);
% Training accuracy 0.967 Test accuracy 0.736
figure(9)
bar([te_acc0 te_acc1 te_acc2 te_acc3])
ylim([0 1]);
ylabel('Test accuracy')
xticklabels({'Pixels', 'H1', 'H2', 'H3'})

% Confusion matrix layer H3:
figure(10)
pred_labels = pred_3 - ones(size(pred_3));
cm3 = confusionchart(true_labels, pred_labels);
cm3.Title = 'Confusion matrix layer H3';
cm3.RowSummary = 'row-normalized';
cm3.ColumnSummary = 'column-normalized';
