library(tidyverse)
library(caret)

set.seed(123)

data <- read.csv("database.csv")

## EDA ##

library(Amelia)
missmap(data)


## DATA CLEANUP ##
library(hms)
data <- data %>% 
  mutate(Date = as.Date(Date, format = "%m/%d/%Y"),
         Time = as.hms(as.character(Time, format = "%H:%M:%S")),
         ID = as.character(ID))


## TRAIN TEST SPLIT ##
index <- createDataPartition(data$Magnitude, p = 0.8, 
                             list = FALSE)
trainData <- data[index, ]
testData  <- data[-index, ]


## MISSINGS FOR TRAIN DATA ##

# drop columns with almost no data
trainData <- trainData %>% 
  select(-Magnitude.Error, -Horizontal.Error)

# select numerics for imputation
toImpute <- trainData %>% 
  select_if(is.numeric)

# get the rest of the variables
trainData <- trainData %>% 
  select_if(negate(is.numeric))

# impute missings using MICE
library(mice)
imputed <- mice(toImpute, method = "norm.boot")
imputed <- complete(imputed)

# recombine dataset
fullTrainData <- cbind(imputed, trainData)

# check that it worked
sum(is.na(fullTrainData))


## CLASS IMBALANCE? ##
# SMOTE



## FEATURE SELECTION ##

# corrplot
library(corrplot)
corrData <- fullTrainData %>% 
  select_if(is.numeric) %>% 
  as.matrix() %>% 
  cor()

corrplot(corrData,
         tl.col = "black", 
         order = "hclust",
         tl.cex = 0.8,
         tl.srt = 70)

# basic LM
model <- lm(Magnitude ~., data = fullTrainData)
summary(model)

# regularization
library(glmnet)
library(mltools)
library(data.table)

oneHot <- one_hot(as.data.table(fullTrainData))

X <- oneHot %>% 
  select(-Magnitude, -Date, -Time, -ID) %>% 
  as.matrix()

Y <- oneHot$Magnitude

lasso <- cv.glmnet(
  x = X,
  y = Y,
  alpha = 1
)

library(coefplot)
extract.coef(lasso, lambda = "lambda.1se")

# plot with names of variables
plot_coeff_evolution = function(regularization, type = 'Lasso'){
  require(ggplot2)
  lambda = regularization$lambda
  coeff = as.matrix(regularization$beta)
  rowName = rownames(coeff)
  coeff = data.table(coeff)
  coeff[ ,name:=rowName]
  coeff = melt(coeff, id.vars = 'name')
  coeff[ ,variable:=rep(lambda, each = length(unique(name)))]
  ggplot(coeff, aes(x = variable, y = value, color = name)) +
    geom_line() +
    xlab('Value of lambda') +
    ylab('Value of coefficient') +
    scale_x_log10() + 
    geom_vline(xintercept = lasso$lambda.1se, linetype = "longdash") + 
    geom_vline(xintercept = lasso$lambda.min, linetype = "longdash") + 
    theme(axis.text = element_text(size = 14),
          axis.title = element_text(size = 14, face = "bold"))
}

plot_coeff_evolution(lasso, "Lasso")

## MODELS ##

# h2o setup
library(h2o)
h2o.init()

trainH2o <- as.h2o(trainData)
testH2o <- as.h2o(testData)

Y <- "Magnitude"
X <- setdiff(names(trainData), Y)

# xgboost
xgb1 <- h2o.xgboost(
  x = X, y = Y, training_frame = trainH2o, ntrees = 5000, learn_rate = 0.05,
  max_depth = 3, min_rows = 3, sample_rate = 0.8, categorical_encoding = "Enum",
  nfolds = 10, fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE, seed = 123, stopping_rounds = 50,
  stopping_metric = "RMSE", stopping_tolerance = 0
)

h2o.performance(xgb1, newdata = trainH2o)

