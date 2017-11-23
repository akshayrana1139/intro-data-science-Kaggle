#########################################################################################
#                                                                                       #
# This scipt is used to load data and pre process and do feature engineering on data    #
# after analysing the relation between data and outputs.                                #
#                                                                                       #
#########################################################################################


# Importing libraries

library(readxl)
library(caret)
library(data.table)
library(knitr)


# Reading data from Excel by specifying data type
train_data <- read_excel("crimeData.xlsx",col_types = c("text","numeric","text","date", "text", "text", "text", "text", "text", "text", "text", "text", "numeric","numeric", "numeric", "numeric", "text", "numeric", "numeric", "numeric", "text", "numeric", "numeric", "text", "text"))
train_data<-na.omit(train_data)

# Removing unwanted columns 
drops <- c("S. No.", "ID", "X Coordinate", "Y Coordinate", "Case Number", "Updated On", "Latitude",
           "Longitude", "Location")
train_data = train_data[ , !(names(train_data) %in% drops)]

# Converting Block names with their last words for easy grouping..
for (i in 1:length(train_data$Block))
{
  a <- strsplit(train_data$Block[i], " ")[[1]]
  train_data$Block[i] <- a[length(a)] 
}

# Converting Date column with month number..
Month <- as.numeric(format(train_data$Date, "%m"))
train_data$Date = Month

# We can use LabelEncoder/Factors or go for One Hot Encoding...


# FACTOR VS ONE_HOT_ENCODING

# There are some cases where LabelEncoder or DictVectorizor are useful, 
# but these are quite limited in my opinion due to ordinality.
# LabelEncoder:  can turn [dog,cat,dog,mouse,cat] into [1,2,1,3,2], 
#    but then the imposed ordinality means that the average of dog and mouse is cat. Still there are algorithms like decision trees and random forests 
#    that can work with categorical variables just fine and LabelEncoder can be used to store values using less disk space.
# One-Hot-Encoding:  has a the advantage that the result is binary rather than ordinal and that everything sits in an orthogonal vector space. 
#   The disadvantage is that for high cardinality, the feature space can really blow up quickly and you start fighting with the curse of dimensionality. 
#   In these cases, I typically employ one-hot-encoding followed by PCA for dimensionality reduction. 

# ONE HOT ENCODING of data. [Converting everything to binary integeres]
# dmy <- dummyVars(" ~ .", data = train_data,fullRank = T)
# train_transformed <- data.frame(predict(dmy, newdata = train_data))
# 16225 X 367
# View(train_transformed)

# FACTOR
factor_vars <- c("Primary Type", "Year", "Severity", "Arrest", "Domestic", "FBI Code", 
                  "Description", "Location Description", "Block", "IUCR", "Beat")
train_data[factor_vars] <- lapply(train_data[factor_vars], function(x) as.numeric(as.factor(x)))
View(train_data)



###################################################################


