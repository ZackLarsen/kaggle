## Kaggle Porto Seguro safe driving competition

#getwd()
setwd('/Users/zacklarsen/Desktop/Kaggle/Safe driving/')


library(data.table)
train <- read.table('train.csv',sep = ',',header = TRUE)

y <- train$target
y[1:5]
#y <- train[,1]
X <- train[,c(0,2:58)]













