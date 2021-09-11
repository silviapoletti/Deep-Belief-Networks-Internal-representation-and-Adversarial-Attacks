% Code provided by Ruslan Salakhutdinov and Geoff Hinton
% Implementation on graphic processors (GPUs) using MATLAB Parallel Computing Toolbox

momentum = init_momentum;

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%
% Visible units are clamped on the data, so we show an input pattern to the
% Restricted Boltzmann Machine and activate the hidden units in order to infer what
% could be the possible features of the input image.
% The result consitutes the new input data for the next layer.
poshidprobs  = 1./(1 + exp(-data_mb * vishid - repmat(hidbiases, numcases, 1))); % activation of hidden units using sigmoid
    % vishid = weight of connections between visible and hidden units
    % hidbiases and visbiases = values of hidden and visible biases
posprods     = data_mb' * poshidprobs;  % correlation between visible and hidden units
poshidact    = sum(poshidprobs); 
posvisact    = sum(data_mb);
%%%%%%%%% END OF POSITIVE PHASE  %%%%%%%%%
poshidstates = poshidprobs > rand(numcases, numhid); % sampling of active unites
                                                     % in order to see which unit has been activated

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%
% Starting from the sampling, we reconstruct (top-down fashion) the visible units, i.e. the input image,
% based on the activation of the hidden units.
negdata     = 1./(1 + exp(-poshidstates * vishid' - repmat(visbiases, numcases, 1)));
neghidprobs = 1./(1 + exp(-negdata * vishid       - repmat(hidbiases, numcases, 1)));
negprods    = negdata' * neghidprobs;   % correlation between visible and hidden units
neghidact   = sum(neghidprobs);
negvisact   = sum(negdata);
%%%%%%%%% END OF NEGATIVE PHASE %%%%%%%%%

err = sqrt(sum(sum((data_mb - negdata).^2)));
if epoch > 5
    momentum = final_momentum;
end

%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%
% Actual learning:
vishidinc  = momentum * vishidinc  + epsilonw*( (posprods-negprods)/numcases - weightcost * vishid);
   % with posprods-negprods, we subtract the correlation of the negative phase 
   % from the the correlation of the positive phase.
   % -> This difference gets smaller over learning time: 
   %    the ability to reconstruct the input image improves after the training
visbiasinc = momentum * visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
hidbiasinc = momentum * hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
vishid     = vishid + vishidinc;
visbiases  = visbiases + visbiasinc;
hidbiases  = hidbiases + hidbiasinc;
%%%%%%%%% END OF UPDATES %%%%%%%%%
gradient = posprods-negprods;
