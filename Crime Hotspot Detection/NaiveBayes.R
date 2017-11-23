###### CALLING CODE TO DEFINE NAIVE BAYES FUNCTIONS AND GET PROCESSED DATA #######
#
source("preprocessing_data.r")
source("NaiveBayesModified.r")
#
##################################################################################


### CLASSIFIER 1 : NAIVE BAYES ###

set.seed(1000)
test <- 1:2000
train <- train_data[-test,]
test<- train_data[test,]

# y has to be a factor to work..
model <- naiveBayesModified(x = train, y = factor(train$Hotspot))
prediction <- predict(model,test)
table(prediction, test$Hotspot)

#Accuracy of the model
confusionMatrix(table(prediction, test$Hotspot))