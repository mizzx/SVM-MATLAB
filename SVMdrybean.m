%%
% Pseudocode
% 1. Read table of dataset 'Dry_Bean_Dataset.xlsx' file
% 2. Specify inputs and targets
% 3. Split the data into 70% training and 30% test
% 4. Create a template for SVM binary classifiers, standardize the predictors, save support vectors and specify kernel function.
% 5. Train a multiclass ECOC classifier using SVM binary learners
% 6. Compute train error and accuracy
% 7. For loop to store support vectors of each SVM
% 8. Scatter plot by group
% 9. Circle the support vectors
% 10. Predict model using test data
% 11. Compute test accuracy
% 12. Compute confusion matrix
% 13. Compute the performance metrics (FPR and TPR)
% 14. Plot ROC curve
% 15. Display the AUC values

%%
% Read table of dataset 'Dry_Bean_Dataset.xlsx' file
drybean = readtable('Dry_Bean_Dataset.xlsx');

% Specify inputs and targets
targets = drybean.Class; 
inputs = drybean(:,1:16); % Select all rows, column of 1 to 16
inputs = table2array(inputs); 

%%
% Use cvpartition to split the data into 70% training and 30% test
n = length(targets);
splitdata = cvpartition(n,'Holdout',0.3); % reserve 30% of the length of the data for test

% Specify xTrain and yTrain
xTrain = inputs(training(splitdata),:);
xTrain = xTrain(:,3:4); 
yTrain = targets(training(splitdata),:);

% Specify xTest and yTest
xTest = inputs(test(splitdata),:);
xTest = xTest(:,3:4); 
yTest = targets(test(splitdata),:);

%%
% Create a template for SVM binary classifiers
% Standardize the predictors, save support vectors and specify kernel function ('gaussian' for rbf / 'linear'/ 'polynomial').
tempSVM = templateSVM('Standardize',true,'SaveSupportVectors',true,'KernelFunction','gaussian');
predictor = {'MajorAxisLength','MinorAxisLength'};
response = 'drybeanTypes';
% Specify class order
class = {'BARBUNYA','BOMBAY','CALI','DERMASON','HOROZ','SEKER','SIRA'};

% Train a multiclass ECOC classifier using SVM binary learners
model = fitcecoc(xTrain,yTrain,'Learners',tempSVM,'ResponseName',response,'PredictorNames',predictor,'ClassNames',class)

%%
% Compute classification loss for multiclass ECOC model
% Compute error and accuracy of training set
trainError = resubLoss(model)
trainAccuracy = 1-trainError

%%
% Use for loop to store support vectors of each SVM
svL = size(model.CodingMatrix,1); % Number of SVMs
sv = cell(svL,1); 
for i = 1:svL
    SVM = model.BinaryLearners{i};
    sv{i} = SVM.SupportVectors;
    sv{i} = sv{i}.*SVM.Sigma + SVM.Mu;
end

%%
% Use gscatter to scatter plot of data by group
figure
MajorAxisLength = xTrain(:,1);  % x-axis values
MinorAxisLength = xTrain(:,2);  % y-axis values
grp = yTrain;   % grouping variable
sym = '.';      % specify marker symbols as '.'
size = 15;      % specify marker size as 15
gscatter(MajorAxisLength,MinorAxisLength,grp,'',sym,size)

%%
% Circle the support vectors
hold on
for i = 1:svL
    svs = sv{i};
    plot(svs(:,1),svs(:,2),'ko','MarkerSize',10);
end
title('Dry Bean - ECOC Support Vectors')
xlabel(predictor{1})
ylabel(predictor{2})
%legend([classNames,'Support Vector'])
legend([class,{'svm'}], 'Location','Best')
hold off

%%
% Predict model using test data set
% Compute test accuracy
predictions = predict(model,xTest);
con = confusionmat(yTest,predictions);
Accuracy = 100*sum(diag(con))/sum(con(:))
%%
% Use confusionchart to compute confusion matrix
ConfMat = confusionchart(yTest,predictions,'RowSummary','total-normalized');
ConfMat.InnerPosition = [0.10 0.12 0.85 0.85];

%%
% Compute the classification scores for the test set
[~,Scores] = predict(model,xTest);

% Compute the performance metrics (FPR and TPR)
% Create a rocmetrics object
% using the true labels in yTest and the classification scores in Scores
% Specify the column order of Scores using Mdl.ClassNames.
roc = rocmetrics(yTest,Scores,model.ClassNames);

% Plot ROC curve by using plot function.
figure
plot(roc)

% Display the AUC values
roc.AUC
