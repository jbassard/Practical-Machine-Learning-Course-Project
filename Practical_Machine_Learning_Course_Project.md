Synopsis
--------

This is the project assignment for the Practical Machine Learning course in Coursera's Data Science specialization track. **The purpose of this project is to create a machine-learning algorithm that can correctly identify how test subjects did the exercice of barbell bicep curls by using data from belt, forearm, arm, and dumbbell accelerometers.**

Original Training Data is available online (<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>) as well as Test data (<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>).

There are five classifications of this exercise, one method is the correct way to do the exercise while the other five are common mistakes: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

### Checking for required packages and install them if necessary, then load them

``` r
if (!require("knitr")) {
    install.packages("knitr")}
```

    ## Loading required package: knitr

``` r
if (!require("caret")) {
    install.packages("caret")}
```

    ## Loading required package: caret

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
if (!require("randomForest")) {
    install.packages("randomForest")}
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

``` r
if (!require("e1071")) {
     install.packages("e1071")}
```

    ## Loading required package: e1071

``` r
if (!require("doParallel")) {
     install.packages("doParallel")}
```

    ## Loading required package: doParallel

    ## Loading required package: foreach

    ## Loading required package: iterators

    ## Loading required package: parallel

``` r
library(knitr)
library(caret)
library(randomForest)
library(e1071)
library(doParallel)
```

### Setting the default of echo and cache to be True throughout the whole report

``` r
knitr::opts_chunk$set(echo = TRUE)
knitr::opts_chunk$set(cache=TRUE)
```

Loading data
------------

### Downloading data in MachineLearning folder

``` r
if(!file.exists("./MachineLearning")) {
    dir.create("./MachineLearning")}
if(!file.exists("./MachineLearning/pml-training.csv")) {
    fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(fileUrl1, destfile="./MachineLearning/pml-training.csv")}
if(!file.exists("./MachineLearning/pml-testing.csv")) {
    fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(fileUrl2, destfile="./MachineLearning/pml-testing.csv")}
```

### Loading the data

``` r
training <- read.csv("./MachineLearning/pml-training.csv", na.strings = c("NA","#DIV/0!", ""))
testing <- read.csv("./MachineLearning/pml-testing.csv", na.strings = c("NA","#DIV/0!", ""))
set.seed(33333) # for reproducibility
```

Quick Exploration of the datasets
---------------------------------

Remark, to limit the lenght of this report outputs generated from this exploratory analysis are not displayed.

### Training set

``` r
dim(training)
head(training)
str(training)
```

Using Dim(), head(), and str() functions, we can see that we have 19622 observations of 160 variables. Severall columns have NAs, and there are irrelevant variables.

### Testing set

``` r
dim(testing)
head(testing)
str(testing)
```

Using Dim(), head(), and str() functions, we can see that as for the training set, several columns have NAs and there are irrelevant variables. In this dataset, we have 20 observations of 160 variables.

Cleaning datasets
-----------------

The data has many variables with missing data as well as information that is not usefull to the question to answer. The columns with these irrelevant data will be erased.

### Training set

``` r
training <- training[, -(1:7)] # remove first 7 variables not needed for prediction (X, user-name,       raw-timestamp_part_1, raw-timestamp_part_2,cvtd_timestamp, new_window, num_window)
NAcolumns <- colnames(testing)[colSums(is.na(testing)) > 0] # remove columns with NAs using testing set as reference
training<-training[,!(names(training) %in% NAcolumns)]
dim(training) # check the final number of variables
```

    ## [1] 19622    53

### Testing set

Same cleaning is applied to the testing set.

``` r
testing <- testing[, -(1:7)] # remove first 7 variables not needed
testing <- testing[,!(names(testing) %in% NAcolumns)]
dim(testing) # check the final number of variables
```

    ## [1] 20 53

Both sets have 53 variables that will be used for the prediction. There are also 19622 observations in the training set and 20 in the testing set.

Preparing the data for the training
-----------------------------------

### Making a trainig and testing subsets

A training subset is created with 75% of the original training dataset to be used for training and the remaining 25% to be used as the testing subset (before final testing is performed with the testing set provided for the project)

``` r
subTrain <- createDataPartition(training$classe, p = .75, list = FALSE)
trainingsubset <- training[subTrain,]
testingsubset <- training[-subTrain,]
dim(testingsubset)
```

    ## [1] 4904   53

``` r
dim(trainingsubset)
```

    ## [1] 14718    53

### Preparing cross-validation test with trainControl

Use of "trainControl" function in caret to do 10 cross-validation tests, and use to select the best model from the 10.

``` r
tc <- trainControl(method = "cv", number = 10, verboseIter=FALSE)
```

Training a random forest model
------------------------------

To generate a predictive model, a random-forest modeling is used with all remaining predictors together with the trainControl ("tc") arguments set up before. Modeling is achieved on 3 processor cores to speed up the process (using the doParallel package).

``` r
detectCores()
```

    ## [1] 8

``` r
getDoParWorkers()
```

    ## [1] 1

``` r
registerDoParallel(cores = 3)
rf_model <- train(classe ~ ., data = trainingsubset, method = "rf", trControl= tc)
```

``` r
rf_model # print the model
```

    ## Random Forest 
    ## 
    ## 14718 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 13246, 13246, 13247, 13248, 13245, 13245, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9930701  0.9912334
    ##   27    0.9942927  0.9927803
    ##   52    0.9892657  0.9864204
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

``` r
varImp(rf_model) #display the importance of variables in the model
```

    ## rf variable importance
    ## 
    ##   only 20 most important variables shown (out of 52)
    ## 
    ##                      Overall
    ## roll_belt            100.000
    ## pitch_forearm         59.394
    ## yaw_belt              55.493
    ## pitch_belt            45.192
    ## roll_forearm          44.382
    ## magnet_dumbbell_z     44.259
    ## magnet_dumbbell_y     42.755
    ## accel_dumbbell_y      21.245
    ## magnet_dumbbell_x     18.091
    ## accel_forearm_x       17.616
    ## roll_dumbbell         15.647
    ## magnet_belt_z         15.379
    ## accel_dumbbell_z      14.372
    ## magnet_forearm_z      13.691
    ## accel_belt_z          13.487
    ## total_accel_dumbbell  13.293
    ## magnet_belt_y         12.501
    ## gyros_belt_z          11.177
    ## yaw_arm               10.828
    ## magnet_belt_x          9.989

The 5 most important variables are roll\_belt, pitch\_forearm, yaw\_belt, pitch\_belt, roll\_forearm.

Evaluation of the model (out-of-sample error)
---------------------------------------------

For evaluation, the model is used to predict the outcome of "testingsubset" generated before from the original training set. Then the function confusionMatrix is used to calculate the accuracy of the prediction.

``` r
predictions <- predict(rf_model, testingsubset)
confusionMatrix(testingsubset$classe,predictions)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1394    1    0    0    0
    ##          B    5  943    1    0    0
    ##          C    0    6  847    2    0
    ##          D    0    0    6  797    1
    ##          E    0    0    1    0  900
    ## 
    ## Overall Statistics
    ##                                         
    ##                Accuracy : 0.9953        
    ##                  95% CI : (0.993, 0.997)
    ##     No Information Rate : 0.2853        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.9941        
    ##  Mcnemar's Test P-Value : NA            
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9964   0.9926   0.9906   0.9975   0.9989
    ## Specificity            0.9997   0.9985   0.9980   0.9983   0.9998
    ## Pos Pred Value         0.9993   0.9937   0.9906   0.9913   0.9989
    ## Neg Pred Value         0.9986   0.9982   0.9980   0.9995   0.9998
    ## Prevalence             0.2853   0.1937   0.1743   0.1629   0.1837
    ## Detection Rate         0.2843   0.1923   0.1727   0.1625   0.1835
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9981   0.9956   0.9943   0.9979   0.9993

``` r
acc <- (max(rf_model$results$Accuracy))*100 # search accuracy value
acc
```

    ## [1] 99.42927

``` r
OSE <- (100-acc) # calculate Out of Sample Error from Accuracy
OSE
```

    ## [1] 0.5707309

The random-forest model has a 99.42% accuracy, thus out-of-sample error is 0.57%, which is really good.

Answering exercise question
---------------------------

The model is used to predict how 20 different test subjects did the exercice of barbell bicep curls (orginal testing dataset provided for the exercise)

``` r
exercise_answers <- predict(rf_model, newdata=testing)
print(exercise_answers)
```

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E

Conclusion
----------

Accuracy achieved with this random-forest model is very high using the testingsubset database. It is important to consider that the samples for this exercise are all taken from one larger sample set. Thus if the data are collected again during a different time period or with different subjects the out of sample error could be higher and the model may not be as accurate.
