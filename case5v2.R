# TODO: remove once package is available on CRAN
library(devtools)
#install_github("gedeck/mlba/mlba", force=TRUE)
library(ModelMetrics)

taxi.df <- mlba::TaxiCancellationCase
head(taxi.df)

## pre-process
# null to zero
taxi.df[is.na(taxi.df)] <- 0

# numbers to factors
taxi.df$vehicle_model_id <- as.factor(taxi.df$vehicle_model_id)
taxi.df$package_id <- as.factor(taxi.df$package_id)
taxi.df$travel_type_id <- as.factor(taxi.df$travel_type_id)

# date to day of week + time
taxi.df$from_date <- as.POSIXlt(taxi.df$from_date,
                             format = "%m/%d/%Y %H:%M")
taxi.df$from_date_DOW <- weekdays(taxi.df$from_date)
taxi.df$from_date_Hour <- taxi.df$from_date$hour
taxi.df$to_date <- as.POSIXlt(taxi.df$from_date,
                              format = "%m/%d/%Y %H:%M")
taxi.df$booking_created <- as.POSIXlt(taxi.df$from_date,
                              format = "%m/%d/%Y %H:%M")
taxi.df$booking_created_DOW <- weekdays(taxi.df$booking_created)

# compute trip length from GPS data
dist <- function (long1, lat1, long2, lat2){
  rad <- pi/180
  a1 <- lat1 * rad
  a2 <- long1 * rad
  b1 <- lat2 * rad
  b2 <- long2 * rad
  dlon <- b2 - a2
  dlat <- b1 - a1
  a <- (sin(dlat/2))^2 + cos(a1) * cos(b1) * (sin(dlon/2))^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  R <- 6378.145
  d <- R * c
  return(d)
}
taxi.df$trip_length <- dist(taxi.df$from_long, taxi.df$from_lat,
                            taxi.df$to_long, taxi.df$to_lat)

#### DATA EXPLORATION ####
# Ensure the date columns are in Date format
taxi.df$from_date <- as.Date(taxi.df$from_date)
taxi.df$to_date <- as.Date(taxi.df$to_date)
taxi.df$booking_created <- as.Date(taxi.df$booking_created)

# Find the minimum and maximum for each date column
range_from_date <- range(taxi.df$from_date, na.rm = TRUE)
range_to_date <- range(taxi.df$to_date, na.rm = TRUE)
range_booking_created <- range(taxi.df$booking_created, na.rm = TRUE)

# Print the ranges
print(paste("Range of from_date:", range_from_date[1], "to", range_from_date[2]))
print(paste("Range of to_date:", range_to_date[1], "to", range_to_date[2]))
print(paste("Range of booking_created:", range_booking_created[1], "to", range_booking_created[2]))

## run prediction methods
library(caret)
library(gains)

t(t(names(taxi.df)))
selected.vars <- c(3:9, 12, 13, 20:23, 19)
set.seed(123)
train.ind <- sample(1:dim(taxi.df)[1], 0.6*dim(taxi.df)[1])
train.df <- taxi.df[train.ind, selected.vars]
valid.df <- taxi.df[-train.ind, selected.vars]

# knn
knnFit <- train(as.factor(Car_Cancellation) ~ .,
                data = train.df,
                method = "knn",
                tuneGrid = expand.grid(.k=c(1, 10)))

pred <- predict(knnFit, valid.df, type = "prob")[,2]
pred_knn <- pred 

## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation,
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred >0.5, 1, 0)),
                as.factor(valid.df$Car_Cancellation))

auc_value<- auc(valid.df$Car_Cancellation,pred_knn)
auc_value

# tree/ random forest
library(rpart)
library(rpart.plot)
library(adabag)

rt <- rpart(Car_Cancellation ~ ., data = train.df)
pfit<- prune(rt, cp=   rt$cptable[which.min(rt$cptable[,"xerror"]),"CP"])
prp(pfit)
pred <- predict(pfit, valid.df)
pred_rt <- pred

## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation,
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred >0.5, 1, 0)),
                as.factor(valid.df$Car_Cancellation))

auc_value<- auc(valid.df$Car_Cancellation,pred_rt)
auc_value

#Boosting
train.df$Car_Cancellation.factor <- as.factor(train.df$Car_Cancellation)
boost <- boosting(Car_Cancellation.factor ~ ., data = train.df[,-14])
pred <- predict(boost, valid.df)$prob[,2]
pred_boost <- pred

## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation,
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred >0.5, 1, 0)),
                as.factor(valid.df$Car_Cancellation))

auc_value<- auc(valid.df$Car_Cancellation,pred_boost)
auc_value

# logistic regression
train.df <- taxi.df[train.ind, selected.vars]
reg <- glm(Car_Cancellation ~ ., data = train.df, family=binomial)
summary(reg)
#pred <- predict(reg, valid.df, type = "response")
# note error: factor vehicle_model_id has new levels 36, 70
# same happens with package_id
# fix levels:
reg$xlevels[["vehicle_model_id"]] <- union(reg$xlevels[["vehicle_model_id"]],
                                           levels(valid.df$vehicle_model_id))
reg$xlevels[["package_id"]] <- union(reg$xlevels[["package_id"]],
                                           levels(valid.df$package_id))
# predict again:
pred <- predict(reg, valid.df, type = "response")
pred_reg <- pred
## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation,
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred >0.5, 1, 0)),
                as.factor(valid.df$Car_Cancellation))

auc_value<- auc(valid.df$Car_Cancellation,pred_reg)
auc_value


#### ENSEMBLE ####
# To be used later in accuracy measurements and fitting ensemble
# models
set.seed(123)
train.actual <- train.df$Car_Cancellation
test.actual <- valid.df$Car_Cancellation

# Getting the training fits (predictions) to be used in
# building some superintendent models for combos
pred_reg_train <- predict(reg,newdata=train.df)
pred_knn_train <- predict(knnFit,newdata=train.df,type = "prob")[,2]
pred_rt_train <- predict(rt,newdata=train.df)
pred_boost_train <- predict(boost,newdata=train.df)$prob[,2]

length(pred_reg_train)
length(pred_knn_train)
length(pred_rt_train)
length(pred_boost_train)

train_ensemble <- data.frame(pred_knn_train, pred_rt_train, pred_boost_train, pred_reg_train, train.actual)
head(train_ensemble)
names(train_ensemble)[names(train_ensemble)=="pred_reg_train"]<-"pr.ols"
names(train_ensemble)[names(train_ensemble)=="pred_knn_train"]<-"pr.knn"
names(train_ensemble)[names(train_ensemble)=="pred_rt_train"]<-"pr.tree"
names(train_ensemble)[names(train_ensemble)=="pred_boost_train"]<-"pr.boost"
names(train_ensemble)[names(train_ensemble)=="train.actual"]<-"Cancellations"
head(train_ensemble)
dim(train_ensemble)

test_ensemble <- data.frame(pred_knn, pred_rt, pred_boost, pred_reg, test.actual)
head(test_ensemble)
names(test_ensemble)[names(test_ensemble)=="pred_reg"]<-"pr.ols"
names(test_ensemble)[names(test_ensemble)=="pred_knn"]<-"pr.knn"
names(test_ensemble)[names(test_ensemble)=="pred_rt"]<-"pr.tree"
names(test_ensemble)[names(test_ensemble)=="pred_boost"]<-"pr.boost"
names(test_ensemble)[names(test_ensemble)=="test.actual"]<-"Cancellations"
head(test_ensemble)
dim(test_ensemble)


ensemble_both <- rbind(train_ensemble,test_ensemble)
head(ensemble_both)
dim(ensemble_both)


ensemble.train <- ensemble_both[train.ind, ]
dim(ensemble.train)
ensemble.test <- ensemble_both[-train.ind,]
dim(ensemble.test)

superintendent_Model <- lm(Cancellations~.,data=ensemble.train)
summary(superintendent_Model)

pr.ensemble.test<-predict(superintendent_Model,newdata=ensemble.test)

gain <- gains(ensemble.test$Cancellations,
              pr.ensemble.test, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(ensemble.test$Cancellations))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(ensemble.test$Cancellations))~c(0, dim(ensemble.test)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pr.ensemble.test > 0.5, 1, 0)),
                as.factor(ensemble.test$Cancellations))

# Calculate ROC curve
auc_value<- auc(ensemble.test[,5], pr.ensemble.test)
auc_value


################################################################################
########################## IMPROVED MODELS #####################################
################################################################################
train.df <- taxi.df[train.ind, selected.vars]

# Handling Data Imbalance
library(smotefamily)
library(groupdata2)
set.seed(123)
# Convert factors to dummy variables, excluding the response variable
train_trans <- data.frame(model.matrix(~ . - 1, data = train.df))  # '-1' to exclude intercept
test_trans <- data.frame(model.matrix(~ . - 1, data = valid.df))  # '-1' to exclude intercept
response <- train.df$Car_Cancellation # Extract the response variable

# Apply SMOTE (using your preferred SMOTE function, ensure it is the correct one)
#data_balanced <- balance(X = train.df, size = 'mean', cat_col = 'Car_Cancellation' )$data[-51] #get rid of class variable

data_balanced <- balance(train_trans, size = 'mean', cat_col = 'Car_Cancellation' )#get rid of class variable


## Custom scoring function that focuses on sensitivity
c_sensitivity <- function(data, lev = NULL, model = NULL) {
  sensitivity <- sensitivity(data[, "pred"], data[, "obs"], lev[1])
  names(sensitivity) <- "sensitivity"
  sensitivity
}

## Custom scoring function that focuses on specificity
c_specificity <- function(data, lev = NULL, model = NULL) {
  specificity <- specificity(data[, "pred"], data[, "obs"], lev[2])
  names(specificity) <- "specificity"
  specificity
}

## TrainControl for Data Imbalance
trainControl <- trainControl(method = "cv",
                             number = 10,
                             summaryFunction = c_specificity,
                             sampling = "smote")  # Include SMOTE for imbalance

## knn
### Knn doesnt directly use SMOTE() but we still train it on transformed data (for ensemble later)
knnFit <- train(as.factor(Car_Cancellation) ~ .,
                data = train_trans,
                method = "knn",
                trControl = trainControl,
                tuneGrid = expand.grid(.k=c(1, 10)))

pred <- predict(knnFit, test_trans, type = "prob")[,2]
pred_knn <- pred 

## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation,
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred >0.5, 1, 0)),
                as.factor(valid.df$Car_Cancellation))

auc_value<- auc(valid.df$Car_Cancellation,pred_knn)
auc_value

# tree/ random forest
#Random Forest
### Calculate class weights inversely proportional to class frequencies
### to handle data imbalance

class_weights <- ifelse(train_trans$Car_Cancellation == "1",
                        sum(train_trans$Car_Cancellation == "0"),
                        sum(train_trans$Car_Cancellation == "1"))

rt <- rpart(Car_Cancellation ~ ., method = "class", weights = class_weights, 
            data = train_trans)
pfit<- prune(rt, cp=  rt$cptable[which.min(rt$cptable[,"xerror"]),"CP"])
prp(pfit)
pred <- predict(pfit, test_trans)[,2] #Predictions of the positive class
pred_rt <- pred

## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation,
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred >0.5, 1, 0)),
                as.factor(valid.df$Car_Cancellation))

auc_value<- auc(valid.df$Car_Cancellation,pred_rt)
auc_value

#Variable importance plot
library(vip)
vip(rt)

#Boosted Model
## BOOST MODEL USES BALANCED DATA
data_balanced$Car_Cancellation.factor <- as.factor(data_balanced$Car_Cancellation)
boost <- boosting(Car_Cancellation.factor ~ ., data = data_balanced[,-50])
pred <- predict(boost, test_trans)$prob[,2]
pred_boost <- pred

## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation,
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred >0.5, 1, 0)),
                as.factor(valid.df$Car_Cancellation))

auc_value<- auc(valid.df$Car_Cancellation,pred_boost)
auc_value

# logistic regression
data_balanced <- data_balanced[,-51]
reg <- glm(Car_Cancellation ~ ., data = data_balanced, family=binomial)
summary(reg)
#pred <- predict(reg, valid.df, type = "response")
# note error: factor vehicle_model_id has new levels 36, 70
# same happens with package_id
# fix levels:
#reg$xlevels[["vehicle_model_id"]] <- union(reg$xlevels[["vehicle_model_id"]],
                                           #levels(valid.df$vehicle_model_id))
#reg$xlevels[["package_id"]] <- #union(reg$xlevels[["package_id"]],
                                     #levels(valid.df$package_id))
# predict again:
pred <- predict(reg, test_trans, type = "response")
pred_reg <- pred

## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation,
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred > 0.5, 1, 0)),
                as.factor(valid.df$Car_Cancellation))

auc_value<- auc(valid.df$Car_Cancellation,pred_reg)
auc_value

#Penalized Logistic Regression
library(glmnet)
# Prepare the matrix of predictors and response variable
x <- model.matrix(Car_Cancellation~ ., data = data_balanced)[,-1]  # Remove intercept
y <- data_balanced$Car_Cancellation

# Grid the lambda constraint from 10^10 to 10^(-2) in equal increments.
grid <- 10^seq(-2, 0, length = 10)

# Define a sequence of alpha values to test
alpha_values <- seq(0, 1, length = 10)  # Test 10 values from 0 (Ridge) to 1 (Lasso)

# Store results
cv_results <- list()

# Loop over alpha values
for(alpha in alpha_values) {
  cv_fit <- cv.glmnet(x, y, family = "binomial", alpha = alpha, lambda = grid)
  cv_results[[paste("Alpha", alpha)]] <- cv_fit
}
# Extract and compare cross-validation errors
cv_errors <- sapply(cv_results, function(fit) min(fit$cvm))

# Find the best alpha based on minimum error
best_alpha_index <- which.min(cv_errors)
best_alpha <- alpha_values[best_alpha_index]

# Get the best model fit at this alpha
best_model <- cv_results[[best_alpha_index]]
best_lambda <- best_model$lambda.min

# Generate predictions
x_valid <- model.matrix(Car_Cancellation ~ ., data = test_trans)[,-1]
pred <- predict(best_model, s = best_lambda, newx = x_valid, type="response")
pred_preg <- pred

## evaluation: run for all methods
gain <- gains(valid.df$Car_Cancellation,
              pred, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(valid.df$Car_Cancellation))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(valid.df$Car_Cancellation))~c(0, dim(valid.df)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pred > 0.5, 1, 0)),
                as.factor(valid.df$Car_Cancellation))

auc_value<- auc(valid.df$Car_Cancellation,pred_preg)
auc_value

#### ENSEMBLE V2 ####
# To be used later in accuracy measurements and fitting ensemble
# models
train.actual <- data_balanced$Car_Cancellation
test.actual <- test_trans$Car_Cancellation

# Getting the training fits (predictions) to be used in
# building some superintendent models for combos
pred_reg_train <- predict(reg,newdata=data_balanced)
pred_knn_train <- predict(knnFit,newdata=data_balanced,type = "prob")[,2]
pred_rt_train <- predict(rt,newdata=data_balanced)[,2]
pred_boost_train <- predict(boost,newdata=data_balanced)$prob[,2]

length(pred_reg_train)
length(pred_knn_train)
length(pred_rt_train)
length(pred_boost_train)

train_ensemble <- data.frame(pred_knn_train, pred_rt_train, pred_boost_train, pred_reg_train, train.actual)
head(train_ensemble)
names(train_ensemble)[names(train_ensemble)=="pred_reg_train"]<-"pr.ols"
names(train_ensemble)[names(train_ensemble)=="pred_knn_train"]<-"pr.knn"
names(train_ensemble)[names(train_ensemble)=="pred_rt_train"]<-"pr.tree"
names(train_ensemble)[names(train_ensemble)=="pred_boost_train"]<-"pr.boost"
names(train_ensemble)[names(train_ensemble)=="train.actual"]<-"Cancellations"
head(train_ensemble)
dim(train_ensemble)

test_ensemble <- data.frame(pred_knn, pred_rt, pred_boost, pred_reg, test.actual)
head(test_ensemble)
names(test_ensemble)[names(test_ensemble)=="pred_reg"]<-"pr.ols"
names(test_ensemble)[names(test_ensemble)=="pred_knn"]<-"pr.knn"
names(test_ensemble)[names(test_ensemble)=="pred_rt"]<-"pr.tree"
names(test_ensemble)[names(test_ensemble)=="pred_boost"]<-"pr.boost"
names(test_ensemble)[names(test_ensemble)=="test.actual"]<-"Cancellations"
head(test_ensemble)
dim(test_ensemble)


ensemble_both <- rbind(train_ensemble,test_ensemble)
head(ensemble_both)
dim(ensemble_both)

train.ind <- sample(1:dim(ensemble_both)[1], 0.6*dim(ensemble_both)[1])
ensemble.train <- ensemble_both[train.ind, ]
dim(ensemble.train)
ensemble.test <- ensemble_both[-train.ind,] #row.names(valid.df)
dim(ensemble.test)
head(ensemble.test)

superintendent_Model <- lm(Cancellations~.,data=ensemble.train)
summary(superintendent_Model)

pr.ensemble.test<-predict(superintendent_Model,newdata=ensemble.test)

gain <- gains(ensemble.test$Cancellations,
              pr.ensemble.test, groups=100)

plot(c(0,gain$cume.pct.of.total*sum(ensemble.test$Cancellations))~c(0,gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l", col = "grey")
lines(c(0,sum(ensemble.test$Cancellations))~c(0, dim(ensemble.test)[1]), lty=2)

confusionMatrix(as.factor(ifelse(pr.ensemble.test > 0.5, 1, 0)),
                as.factor(ensemble.test$Cancellations))

# Calculate ROC curve & AUC value
auc_value<- auc(ensemble.test[,5], pr.ensemble.test)
auc_value






