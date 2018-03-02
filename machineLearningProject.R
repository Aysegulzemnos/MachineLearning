.libPaths()  

install.packages("rpart.plot")
install.packages("rpart")
install.packages('fancyRpartPlot',repos='http://cran.us.r-project.org')
install.packages("fancyRpartPlot")
install.packages("caret", dependencies = c("Depends", "Imports","Suggests"))
install.packages("caret") 
install.packages("yaml")
install.packages("rattle")
install.packages('ggplot2');

library(ggplot2);

library(knitr)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(caret)
library(rpart)
library(RColorBrewer)
library(gbm)
library(plyr)
library(rpart)
library(dplyr)



##--------------------------------


dt_training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
dt_testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))


options <- names(dt_testing[,colSums(is.na(dt_testing)) == 0])[8:59]

# Only use features used in testing cases.
dt_training <- dt_training[,c(options,"classe")]
dt_testing <- dt_testing[,c(options,"problem_id")]

dim(dt_training); dim(dt_testing);


##Number of records per classe
library(dplyr)
library(caret)
summarize(group_by(dt_training , classe), dt_training =n())

str(dt_training)


##Split data
##Partitioning the Dataset
###Following the recommendation in the course Practical Machine Learning, we will split our data into a training data set (60% of the total cases) and a testing data set (40% of the total cases; the latter should not be confused with the data in the pml-testing.csv file). This will allow us to estimate the out of sample error of our predictor.



set.seed(54321)

inTrain <- createDataPartition(dt_training$classe, p=0.6, list=FALSE)
training <- dt_training[inTrain,]
testing <- dt_training[-inTrain,]
dim(training); dim(testing);

##Models

##---------------------------------------------------------------
###1.Classification Tree Model

set.seed(11204)
modelTree <- train(classe~., data = training[-1], method="rpart")
modelTree

plot(modelTree)

class(modelTree)
dim(str(training[-1]))

###The accuracy is only 53%. 

predTree <- predict(modelTree, newdata =testing)
table(predTree,testing$classe)
qplot(predTree)


TruePredictionTree <- data.frame(classe=testing$classe,esi=predTree==testing$classe)
frequenzeTree <- table(TruePredictionTree$classe,TruePredictionTree$esi)
frequenzeTree

prop.table(frequenzeTree,1)*100
colSums(frequenzeTree)
colSums(frequenzeTree)/sum(frequenzeTree)

###As expected seeing the accuracy, not very well prediction,only about 50% true. Very bad in particular for classe D.

##====================================================================
###2.Building the Random Forest Model

###Using random forest, the out of sample error should be small. The error will be estimated using the 40% testing sample. We should expect an error estimate of < 3%.

set.seed(11204)
modelRF <- train(classe~., data = training[-1], method="rf", trControl = trainControl(method = "cv", classProbs=TRUE,savePredictions=TRUE,allowParallel=TRUE, number = 10))
modelRF
plot(modelRF)

###I test the Random Forest model.

predRF <- predict(modelRF, newdata = testing)
table(predRF,testing$classe)


TruePredictionRF <- data.frame(classe=testing$classe,esit=predRF==testing$classe)
frequenzeRF <- table(TruePredictionRF$classe,TruePredictionRF$esit)
frequenzeRF

prop.table(frequenzeRF,1)*100

colSums(frequenzeRF)

colSums(frequenzeRF)/sum(frequenzeRF)

###Random forest give us a very better prediction, 99% true: So I choose this model for the final predictions.

##====================================================================
###3.Building the Decision Tree Model

###Now I use the training data for building the model
###Using Decision Tree, we shouldn't expect the accuracy to be high. In fact, anything around 80% would be acceptable.

##library(rpart.plot)
##library(RColorBrewer)
##library(rattle)


set.seed(11204)
modFitDT <- rpart(classe ~ ., data = training, method="class", control = rpart.control(method = "cv", number = 10))
fancyRpartPlot(modFitDT)


##Predicting with the Decision Tree Model

set.seed(112041)

prediction <- predict(modFitDT, testing, type = "class")
confusionMatrix(prediction, testing$classe)

##=====================================================================
##4.Building the Boosting Model

modFitBoost <- train(classe ~ ., method = "gbm", data = training,
                     verbose = F,
                     trControl = trainControl(method = "cv", number = 10))


plot(modFitBoost)

##=======================================================================
##Prediction

##I make the prediction with the Random Forest model on the 20 cases provided.

str(predict(modelRF, newdata = testing))

##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E

20*0.994

## [1] 19.88

I expect to have only 0.6% of error, so 20 on 20 or at least 19 on 20 of correct predictions.
(PS: The predictions on the test are 100% corrects)


##==========================================================================

##Predicting with the Random Forest Model

prediction <- predict(modFitRF, testing, type = "class")
confusionMatrix(prediction, testing$classe)

###The random forest model performed very well in-sample, with about 99.3% Accuracy.


##Predicting with the Boosting Model

prediction <- predict(modFitBoost, testing)
confusionMatrix(prediction, testing$classe)


##Predicting with the Testing Data (pml-testing.csv)

###Decision Tree Prediction

predictionDT <- predict(modFitDT, dt_testing)
predictionDT

###Random Forest Prediction

predictionRF <- predict(modFitRF, dt_testing)
predictionRF

###Boosting Prediction

predictionBoost <- predict(modFitBoost, dt_testing)
predictionBoost



Submission file

As can be seen from the confusion matrix the Random Forest model is very accurate, about 99%. Because of that we could expect nearly all of the submitted test cases to be correct. It turned out they were all correct.



