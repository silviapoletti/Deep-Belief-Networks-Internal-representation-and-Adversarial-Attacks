%%%%%%%%%%%%%%%%%%
% EMNIST LETTERS %
%%%%%%%%%%%%%%%%%%

clear all; close all;

%% NETWORK SETUP

DN.layersize   = [100,200,500];         % network architecture
DN.nlayers     = length(DN.layersize);
DN.maxepochs   = 30;                    % unsupervised learning epochs
DN.batchsize   = 120;                   % mini-batch size
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
fname = 'emnist-letters.mat';
load(fname);

% dataset is a structure containing train and test structures:
inputdata = dataset.train.images;      % matrix 124800x784
dataindexes = dataset.train.labels;    %  array 124800x1
testinputdata = dataset.test.images;   % matrix  20800x786
testdataindexes = dataset.test.labels; %  array  20800x1    

% convert integers into double format and normalize:
inputdata = im2double(inputdata);
dataindexes = im2double(dataindexes);
testinputdata = im2double(testinputdata);
testdataindexes = im2double(testdataindexes);

% some examples from the dataset:
% %for i = 1:10
%     test_img = inputdata(i,:);                  % get the i-th input
%     test_img = reshape(test_img,[28 28]);       % reshape the array into a matrix
%     imshow(test_img,[]);
%     caption = sprintf('EMNIST Letters\n label %d',dataindexes(i));
%     title(caption, 'FontSize', 10);
%     drawnow;
%     pause(0.5);
% %end

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
%2.464219e+02 
fname = 'DBN_emnist_letters.mat';
save(fname, 'DN');

%% RECEPTIVE FIELDS

fname = 'DBN_emnist_letters.mat';
load(fname)

figure(1)
plot_L1(DN, 50) 
figure(2)
plot_L2(DN, 50) 
figure(3)
plot_L3(DN, 50)

%% MODEL EVALUATION

% one-hot encoding:
tr_labels = zeros(size(inputdata,1),26);
for i = 1:size(inputdata,1)
    x = dataindexes(i);
    tr_labels(i,x)=1;
end  

te_labels = zeros(size(testinputdata,1),26);
for i = 1:size(testinputdata,1)
    x = testdataindexes(i);
    te_labels(i,x)=1;
end  

tr_patt = inputdata;
te_patt = testinputdata;

% learning analisys:
fprintf('\nGive as input to the classifier the raw images...\n');
[W0, tr_acc0, te_acc0] = perceptron(tr_patt, tr_labels, te_patt, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc0, te_acc0);
% Training accuracy 0.593 Test accuracy 0.579

fprintf('\nGive as input to the classifier the hidden activations of layer 1..\n');
H1_tr = 1./(1 + exp(-tr_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(tr_patt,1),1)));
H1_te = 1./(1 + exp(-te_patt*DN.L{1}.vishid - repmat(DN.L{1}.hidbiases, size(te_patt,1),1)));
[W1, tr_acc1, te_acc1] = perceptron(H1_tr, tr_labels, H1_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc1, te_acc1);
% Training accuracy 0.622 Test accuracy 0.622

fprintf('\nGive as input to the classifier the hidden activations of layer 2..\n');
H2_tr = 1./(1 + exp(-H1_tr*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_tr,1),1)));
H2_te = 1./(1 + exp(-H1_te*DN.L{2}.vishid - repmat(DN.L{2}.hidbiases, size(H1_te,1),1)));
[W2, tr_acc2, te_acc2] = perceptron(H2_tr, tr_labels, H2_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc2, te_acc2);
% Training accuracy 0.679 Test accuracy 0.676

fprintf('\nGive as input to the classifier the hidden activations of layer 3..\n');
H3_tr = 1./(1 + exp(-H2_tr*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_tr,1),1)));
H3_te = 1./(1 + exp(-H2_te*DN.L{3}.vishid - repmat(DN.L{3}.hidbiases, size(H2_te,1),1)));
[W3, tr_acc3, te_acc3] = perceptron(H3_tr, tr_labels, H3_te, te_labels);
fprintf(1,'Training accuracy %.3f Test accuracy %.3f\n', tr_acc3, te_acc3);
% Training accuracy 0.812 Test accuracy 0.810

figure(4)
bar([te_acc0 te_acc1 te_acc2 te_acc3])
ylim([0.4 1]);
ylabel('Test accuracy')
xticklabels({'Pixels', 'H1', 'H2', 'H3'})
figure(5)
bar([tr_acc0 tr_acc1 tr_acc2 tr_acc3])
ylim([0.4 1]);
ylabel('Train accuracy')
xticklabels({'Pixels', 'H1', 'H2', 'H3'})

%% FEATURE VISUALIZATION ON PYTHON

L.first = H1_tr(1:5000,:);
L.second = H2_tr(1:5000,:);
L.third = H3_tr(1:5000,:);
L.raw = inputdata(1:5000,:);
L.labels = dataindexes(1:5000);
fname = 'features_letters.mat';
save(fname, 'L');
