library(rpart)
library(readxl)
source("preprocessing_data.r")

set.seed(1000)
test <- 1:2000
training <- train_data[-test,]
testing <- train_data[test,]

# grow tree 
fit = rpart(formula = Hotspot ~ Date + Block	+ IUCR	+ `Primary Type`	+ Description	+ 
              Severity	+ `Location Description`	+ Arrest	+ Domestic	+ Beat	+ District	+ Ward	+ 
              `Community Area`	+ `FBI Code`	+ Year + Severity + Arrest, 
            data=training, method="class", parms = list(split="gini"), 
            control = rpart.control(xval = 5, minsplit = 1, cp = 0))

printcp(fit) # display the results 
summary(fit) # detailed summary of splits

library(rattle)
library(rpart.plot)
library(RColorBrewer)

fancyRpartPlot(fit)

Prediction <- predict(fit, testing, type = "class")
mtab<-table(Prediction,testing$Hotspot)
confusionMatrix(mtab)
