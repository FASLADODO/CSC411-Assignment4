load labeled_images
init_test_data

fprintf('Starting to train svm')
%%%SVM with Gabor%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Apply Gabor Energy Filters
test_images = gabor_features(test_images);
tr_images = gabor_features(tr_images);

%---------------------------------
nfold = 5;
ntr = size(tr_images, 1);

if (~exist('tr_identity', 'var'))
  % random permutation (disregards similar faces)
  perm = randperm(ntr); 

  foldsize = floor(ntr/nfold);
  for i=1:nfold-1
    foldids{i} = (i-1)*foldsize+1:(i*foldsize);
  end
  foldids{nfold} = (nfold-1)*foldsize+1:ntr;
else
  % generally one uses random permutation to specify the splits, but because of the special structure of the dataset
  % we use the identity of poeple for this purpose.
  unknown = find(tr_identity == -1);
  tr_identity(unknown) = -(1:length(unknown));
  
  % finding people with the same identity
  [sid ind] = sort(tr_identity);
  [a b] = unique(sid);
  npeople = length(a);

  % separating out people with the same identity
  people = cell(npeople,1);
  people{1} = ind(1:b(1));
  for i=2:npeople
    people{i} = ind(b(i-1)+1:b(i))';
  end
  
  % shuffling people
  people = people(randperm(npeople));
  
  % dividing people into groups of roughly the same size but not necessarily
  foldsize = floor(npeople/nfold);
  for i=1:nfold-1
    foldids{i} = [people{(i-1)*foldsize+1:(i*foldsize)}];
  end
  foldids{nfold} = [people{(nfold-1)*foldsize+1:npeople}];
end

% perform nfold training and validation
for i=1:nfold
  traini_ids = [foldids{[1:(i-1) (i+1):nfold]}];
  testi_ids = foldids{i};

  predi = svm(tr_images(traini_ids, :), tr_labels(traini_ids), tr_images(testi_ids, :));
  
  % display([predi'; tr_labels(testi_ids)']);
  
  acc(i) = sum(predi == tr_labels(testi_ids))/length(foldids{i});
end

mean_acc = mean(acc);
prediction_svm = svm(tr_images, tr_labels, test_images);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load labeled_images
init_test_data
fprintf('Done training SVM, starting to train multiclass');
%%%%Multiclass logistic%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prediction_multiclass = multiclass_logistic_classifier(0, tr_images, tr_labels, test_images);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load labeled_images
init_test_data

fprintf('Done training multiclass, starting to train neural net')
%%%%Neural Net with Gabor%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tr_images = gabor_features(tr_images);
tr_images = tr_images';
tr_labels = one_hot_encode_labels(tr_labels, 7);
tr_labels = tr_labels';

%change the values in this script to edit the parameters for the neural net
hiddenLayerSize = 500;
neural_net_script_2;

test_images = gabor_features(test_images);
test_images = test_images';
test_outputs = net(test_images);
prediction_nn = make_prediction_from_prob(test_outputs');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Done training nn, now predicting');

prediction_ensemble = zeros(size(prediction_svm,1), 3);
prediction_ensemble(:,1) = prediction_svm;
prediction_ensemble(:,2) = prediction_multiclass;
prediction_ensemble(:,3) = prediction_nn;

prediction = zeros(size(prediction_ensemble, 1));
for j = 1:size(prediction_ensemble,1)
    prediction(j) = mode(prediction_ensemble(j,:));
end