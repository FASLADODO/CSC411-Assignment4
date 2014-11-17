load labeled_images.mat;
load public_test_images.mat;

tr_image = reshape_data(tr_images);
test_images = reshape_data(public_test_images);

ensemble = fitensemble(tr_image', tr_labels, 'AdaBoostM2', 50, 'Tree');   
predict = predict(ensemble, test_images');

generate_prediction_submission(predict);