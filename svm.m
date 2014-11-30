function [ prediction ] = svm(tr_images, tr_labels, test_images)

% Create label vectors for binary classification
n = numel(tr_labels);
class1_labels = zeros(n,1);
class2_labels = zeros(n,1);
class3_labels = zeros(n,1);
class4_labels = zeros(n,1);
class5_labels = zeros(n,1);
class6_labels = zeros(n,1);
class7_labels = zeros(n,1);

for i = 1:n
    if tr_labels(i) == 1
        class1_labels(i) = 1;
    elseif tr_labels(i) == 2
        class2_labels(i) = 1;
    elseif tr_labels(i) == 3
        class3_labels(i) = 1;
    elseif tr_labels(i) == 4
        class4_labels(i) = 1;
    elseif tr_labels(i) == 5
        class5_labels(i) = 1;
    elseif tr_labels(i) == 6
        class6_labels(i) = 1;
    elseif tr_labels(i) == 7
        class7_labels(i) = 1;
    end
end

SVMModel_class1 = fitcsvm(tr_images, class1_labels, 'BoxConstraint', 1e-2, 'KernelScale', 1);
% CVSVMModel_class1 = crossval(SVMModel_class1);
% kfoldLoss(CVSVMModel_class1);

SVMModel_class2 = fitcsvm(tr_images, class2_labels, 'BoxConstraint', 1e-2, 'KernelScale', 1);
% CVSVMModel_class2 = crossval(SVMModel_class2);
% kfoldLoss(CVSVMModel_class2);

SVMModel_class3 = fitcsvm(tr_images, class3_labels, 'BoxConstraint', 1e-2, 'KernelScale', 1);
% CVSVMModel_class3 = crossval(SVMModel_class3);
% kfoldLoss(CVSVMModel_class3);

SVMModel_class4 = fitcsvm(tr_images, class4_labels, 'BoxConstraint', 1e-2, 'KernelScale', 1);
% CVSVMModel_class4 = crossval(SVMModel_class4);
% kfoldLoss(CVSVMModel_class4);

SVMModel_class5 = fitcsvm(tr_images, class5_labels, 'BoxConstraint', 1e-2, 'KernelScale', 1);
% CVSVMModel_class5 = crossval(SVMModel_class5);
% kfoldLoss(CVSVMModel_class5);

SVMModel_class6 = fitcsvm(tr_images, class6_labels, 'BoxConstraint', 1e-2, 'KernelScale', 1);
% CVSVMModel_class6 = crossval(SVMModel_class6);
% kfoldLoss(CVSVMModel_class6);

SVMModel_class7 = fitcsvm(tr_images, class7_labels, 'BoxConstraint', 1e-2, 'KernelScale', 1);
% CVSVMModel_class7 = crossval(SVMModel_class7);
% kfoldLoss(CVSVMModel_class7);

ScoreSVMModel_class1 = fitPosterior(SVMModel_class1,tr_images,class1_labels);
ScoreSVMModel_class2 = fitPosterior(SVMModel_class2,tr_images,class2_labels);
ScoreSVMModel_class3 = fitPosterior(SVMModel_class3,tr_images,class3_labels);
ScoreSVMModel_class4 = fitPosterior(SVMModel_class4,tr_images,class4_labels);
ScoreSVMModel_class5 = fitPosterior(SVMModel_class5,tr_images,class5_labels);
ScoreSVMModel_class6 = fitPosterior(SVMModel_class6,tr_images,class6_labels);
ScoreSVMModel_class7 = fitPosterior(SVMModel_class7,tr_images,class7_labels);


% Generate soft prediction scores
[~, score1] = predict(ScoreSVMModel_class1, test_images);
[~, score2] = predict(ScoreSVMModel_class2, test_images);
[~, score3] = predict(ScoreSVMModel_class3, test_images);
[~, score4] = predict(ScoreSVMModel_class4, test_images);
[~, score5] = predict(ScoreSVMModel_class5, test_images);
[~, score6] = predict(ScoreSVMModel_class6, test_images);
[~, score7] = predict(ScoreSVMModel_class7, test_images);

all_score = zeros(size(test_images,1),7);
all_score(:,1) = score1(:,2);
all_score(:,2) = score2(:,2);
all_score(:,3) = score3(:,2);
all_score(:,4) = score4(:,2);
all_score(:,5) = score5(:,2);
all_score(:,6) = score6(:,2);
all_score(:,7) = score7(:,2);

prediction = make_prediction_from_prob(all_score);
