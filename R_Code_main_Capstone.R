# Setting up the working directory
setwd("D:/Krishna Great lakes/Capstone")

# Reading the data set 
my_data = read.csv("capstone_dataset.csv", header = TRUE)

# Names of the dependent and independent variables
names(my_data)

# The dimensions of my data
dim(my_data)

# Splitting the data into training and test sets
set.seed(1234)
library(caTools)
sample = sample.split(my_data, SplitRatio = 0.7)
training_set = subset(my_data, sample == TRUE)

# Checking the type of variables 
str(training_set)

# Converting the dependent integar variable  to factor as it is also a category
training_set$click = as.factor(training_set$click)

# Converting the independent integar variables to factor variables as they are categorical 
training_set$banner_pos       = as.factor(training_set$banner_pos)
training_set$site_id          = as.factor(training_set$site_id)
training_set$site_domain      = as.factor(training_set$site_domain)
training_set$site_category    = as.factor(training_set$site_category)
training_set$app_id           = as.factor(training_set$app_id)
training_set$app_domain       = as.factor(training_set$app_domain)
training_set$app_category     = as.factor(training_set$app_category)
training_set$device_type      = as.factor(training_set$device_type)
training_set$device_conn_type = as.factor(training_set$device_conn_type)


# Re-checing the variables in order to make sure that the variables are converted  
str(training_set)

# Checking the balance in the data set
table(training_set$click)

# SMOTEing the data set in order to balance the dataset
library(lattice)
library(grid)
library(DMwR)

training_set_SMOTE = SMOTE(click ~ ., training_set, perc.over = 200)

table(training_set_SMOTE$click)

# Checking the levels of the factor variables in the dataset
levels(training_set_SMOTE$click)
levels(training_set_SMOTE$banner_pos)
levels(training_set_SMOTE$site_id)
levels(training_set_SMOTE$site_domain)
levels(training_set_SMOTE$site_category)
levels(training_set_SMOTE$app_id)
levels(training_set_SMOTE$app_domain)
levels(training_set_SMOTE$app_category)
levels(training_set_SMOTE$device_type)
levels(training_set_SMOTE$device_conn_type)

# Checking for missing values  
which(is.na(training_set_SMOTE))


# Applying the logistic regression model on the training set
classifier = glm(click ~ ., data = training_set_SMOTE, family = binomial)
summary(classifier)


# Calculating the probability for the training set
prob_pred = predict(classifier, type = "response", newdata = training_set_SMOTE[,-1])
y_pred = ifelse(prob_pred > 0.5, 1, 0)
 
#Making the confusion matrix
library(InformationValue)
confusionMatrix(training_set_SMOTE[,1], y_pred)

specificity(training_set_SMOTE$click, y_pred)
precision(training_set_SMOTE$click, y_pred)
# Plotting the ROC curve in order to find the area under the curve
library(Deducer)
rocplot(classifier)

# Reading the test data where we need to test our model
my_test_data = read.csv("capstone_test_dataset.csv", header = TRUE)

# The dimensions of the test data
dim(my_test_data)

# Checking the names 
my_test_data = my_test_data[-1]
names(my_test_data)

# Checking the data types
str(my_test_data)

# Converting all the variables to factors as they are categorical
my_test_data$click            = as.factor(my_test_data$click)
my_test_data$banner_pos       = as.factor(my_test_data$banner_pos)
my_test_data$site_id          = as.factor(my_test_data$site_id)
my_test_data$site_domain      = as.factor(my_test_data$site_domain)
my_test_data$site_category    = as.factor(my_test_data$site_category)
my_test_data$app_id           = as.factor(my_test_data$app_id)
my_test_data$app_domain       = as.factor(my_test_data$app_domain)
my_test_data$app_category     = as.factor(my_test_data$app_category)
my_test_data$device_type      = as.factor(my_test_data$device_type)
my_test_data$device_conn_type = as.factor(my_test_data$device_conn_type)

# Cheking the balance in the data set
table(my_test_data$click)

# SMOTEing the data set in order to balance the dataset
library(lattice)
library(grid)
library(DMwR)

test_set_SMOTE = SMOTE(click ~ ., my_test_data, perc.over = 200)

table(training_set_SMOTE$click)
str(training_set)
View(training_set_SMOTE)


# Predicting the test results
prob_pred = predict(classifier, type = "response", newdata = my_test_data[,-1])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the confusion matrix
cm_test = confusionMatrix(my_test_data[,1], y_pred)
cm_test

specificity(my_test_data[,1], y_pred)
precision(my_test_data[,1], y_pred)

library(lmtest)
install.packages("pscl")
library(pscl)
lrtest(classifier)
pR2(classifier)
# Applying the random forest classification 
library(randomForest)
set.seed(1234)
class(training_set_SMOTE$click)
training_set_SMOTE$click = as.factor(training_set_SMOTE$click)
classifier_RF = randomForest(click ~ ., data = training_set_SMOTE, 
                             ntree=100, mtry = 3, nodesize = 180,
                             importance=TRUE)
plot(classifier_RF) 


set.seed(1234)
T2_RF = tuneRF(x = training_set_SMOTE[,-1], 
               y= training_set_SMOTE$click,
               mtryStart = 3,
               ntreeTry=100, 
               stepFactor = 1, ## 1st try 2 variables, next 4 , next 5 , next 6 MtryStart*Stepfactor 
               improve = 0.001, ## delta OOB 
               trace=TRUE, 
               plot = TRUE,
               doBest = TRUE,
               nodesize = 180, 
               importance=TRUE
) 
# random Forest tuning also lead to mtry = 2
T2_RF$mtry
y_pred = predict(T2_RF, newdata = training_set_SMOTE[-1], type='class')
CM = table(training_set_SMOTE$click, y_pred)
CM
Specificity = CM[1]/(CM[1]+CM[3])
Presision = CM[4]/(CM[4]+CM[3])

# y_pred = predict(T2_RF, newdata = my_test_data[-1], type='class')
# CM = table(my_test_data$click, y_pred)
# CM
# Specificity = CM[1]/(CM[1]+CM[3])
# Presision = CM[4]/(CM[4]+CM[3])

y_pred = predict(T2_RF, newdata = test_set_SMOTE[-1], type='class')
CM = table(test_set_SMOTE$click, y_pred)
CM
Specificity = CM[1]/(CM[1]+CM[3])
Presision = CM[4]/(CM[4]+CM[3])


CM = confusionMatrix(training_set_SMOTE$click, y_pred, threshold  = 0.5)
str(y_pred)
training_set_SMOTE$click = as.numeric(training_set_SMOTE$click)
y_pred1 = as.numeric(y_pred)
y_pred 
CM
T2_RF$importance
training_set_SMOTE$click = as.numeric(training_set_SMOTE$click)

sapply(my_test_data, class)
y_pred = predict(T2_RF, newdata = test_set_SMOTE[-1], type='class')
CM = confusionMatrix(test_set_SMOTE$click, y_pred)
CM
T2_RF$importance

# Applying the random forest classification 
library(randomForest)
set.seed(1234)
classifier_RF = randomForest(click ~ ., data = training_set, 
                             ntree=500, mtry = 3, nodesize = 180,
                             importance=TRUE)
plot(classifier_RF) 

classifier_RF = randomForest(click ~ ., data = training_set, 
                             ntree=50, mtry = 3, nodesize = 100,
                             importance=TRUE)
plot(classifier_RF) 

set.seed(1234)
T2_RF = tuneRF(x = training_set[,-1], 
               y= training_set$click,
               mtryStart = 3,
               ntreeTry=50, 
               stepFactor = 1, ## 1st try 2 variables, next 4 , next 5 , next 6 MtryStart*Stepfactor 
               improve = 0.001, ## delta OOB 
               trace=TRUE, 
               plot = TRUE,
               doBest = TRUE,
               nodesize = 180, 
               importance=TRUE
) # random Forest tuning also lead to mtry = 2
T2_RF$mtry
y_pred = predict(T2_RF, newdata = training_set[-1], type='class')
CM = confusionMatrix(training_set$click, y_pred1)
training_set$click = as.numeric(training_set$click)
y_pred1 = as.numeric(y_pred)
?confusionMatrix
CM
T2_RF$importance

sapply(my_test_data, class)
y_pred = predict(T2_RF, newdata = my_test_data[-1], type='class')
CM = confusionMatrix(my_test_data$click, y_pred)
CM
T2_RF$importance

