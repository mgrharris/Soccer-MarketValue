##############################################
##### BAX-452 ################################
##### January 27, 2021 #######################
##### Assignment 3 ###########################
### Michael Harris, Qihan Guan, Chloe Wang ###
##############################################


### Load Data ###

data <- read.csv("FIFA_Player_List.csv")
data <- na.omit(data)
head(data,10)
names(data)
str(data)
summary(data)
lapply(data, class)

#########################################
####### Q1. Data Analysis ##############
###########################################

library(tidyr)
library(ggplot2)
library(reshape2)
library(corrplot)
library(dplyr)
library(purrr)

data %>%
  keep(is.numeric) %>%                     
  gather() %>%                            
  ggplot(aes(value)) +                     
  facet_wrap(~ key, scales = "free") +   
  geom_histogram(bins=12)


#Count plot for the categorical variable preferred foot
ggplot(data) + geom_bar(aes(x = Preferred.Foot))

# Correlation Heatmap of independent numeric variables 
cor <- cor(data[,unlist(lapply(data, is.numeric))])
melted_cor <- melt(cor)
ggplot(data = melted_cor, aes(x=Var1, y=Var2, fill=value)) + geom_tile() + scale_x_discrete(guide = guide_axis(angle = 90))

# Correlation Plot of independent numeric variables 

corrplot(cor, method = 'circle')

# One issue: High correlations between different skills (passing, shooting, overall.score, mental,
# ball skills, defense, etc.) It is helpful to resolve multicollinearity. 

# We can also explore interactions between the variables. 


#height*defence

inter1 <- lm(data$Market.Value~data$Height*data$Ball.Skills)
summary(inter1)

#age*physical
inter2 <- lm(data$Market.Value~data$Age*data$Physical)
summary(inter2)

#weight*physical
inter3 <- lm(data$Market.Value~data$Weight*data$Physical)
summary(inter3)

#ball.skills*shooting
inter4 <- lm(data$Market.Value~data$Ball.Skills*data$Shooting)
summary(inter4)

#preferred.foot*shooting
inter5 <- lm(data$Market.Value~factor(data$Preferred.Foot)*data$Shooting)
summary(inter5)
interaction.plot(factor(data$Preferred.Foot), data$Shooting, data$Market.Value)

#preferred.foot*passing
inter6 <- lm(data$Market.Value~factor(data$Preferred.Foot)*data$Passing)
summary(inter6)
interaction.plot(factor(data$Preferred.Foot), data$Passing, data$Market.Value)


#preferred.foot*scoring
inter7 <- lm(data$Market.Value~factor(data$Preferred.Foot)*data$Overall.Score)
summary(inter7)
interaction.plot(factor(data$Preferred.Foot), data$Overall.Score, data$Market.Value)


#preferred.foot*ball skills
inter8 <- lm(data$Market.Value~factor(data$Preferred.Foot)*data$Ball.Skills)
summary(inter8)
interaction.plot(factor(data$Preferred.Foot), data$Ball.Skills, data$Market.Value)



#####################################
##### Q2. Feature Selection##########
#####################################

#Randomly split data into train set and test set (80% training, 20% testing)

set.seed(10)

data_copy = data

size_train = floor(0.8*nrow(data_copy))
size_test = floor(0.2*nrow(data_copy))
i_train    <- sort(sample(seq_len(nrow(data_copy)), size=size_train))
i_train_exclude <- setdiff(seq_len(nrow(data_copy)), i_train)
i_test  <- sort(sample(i_train_exclude, size=size_test))

data_train <- data_copy[i_train,]
data_test <- data_copy[i_test,]

#Make copies in case of modification
data_train_copy <- data_train
data_test_copy <- data_test

#Create dummy variables
data_train_copy$foot_dummy <- ifelse(data_train_copy$Preferred.Foot=='Right',1,0)
data_train_copy$Preferred.Foot <- NULL

#First let's consider a full model
model1 <- lm(Market.Value ~ Overall.Score + Potential.Score + Weekly.Salary + Height + Weight 
             + Age + Ball.Skills + Defence + Mental + Passing + Physical + Shooting
             + Goalkeeping + foot_dummy, data=data_train_copy)
summary(model1)

#-----------------------------------------------------------------------------------------------------------
# Determine whether to drop some features 
# Correlations vs. our Dependent Variable. 
library(dplyr)
numeric <- select_if(data, is.numeric)
marketval_cor <- cor(data$Market,numeric)
marketval_cor[,abs(marketval_cor)>=0.1]
marketval_cor[,abs(marketval_cor)<0.1]

# We may consider to drop the variables that have little(<0.1) correlation with the dependent variable. 
# However, we should note from the interaction analysis that both 'Age' and 'Weight' interacts with 'Physical', 
# and 'Physical' shows some a fair degree of correlation with the dependent variable. 
# Therefore, we will not drop 'Age' or 'Weight'. 
# Moreover, 'Goalkeeping' interacts with 'Defence' and 'Height' interacts with 'Ball.skills'. 
# We choose not to drop 'Goalkeeping' or 'Height'
# By far, we choose not to drop any features. 

#-------------------------------------------------------------------------------------------------

# Stepwise Regression for feature selection
# Even though we currently do not have rationale to drop features, 
# we can still use Stepwise Regression to find the optimal selection of features.

library(olsrr)

steps<-ols_step_both_p(model1)

#Summary
steps

#Predictors
steps$predictors

#model
summary(steps$model)

#Plot
par(mfrow=c(1,1))
ols_plot_obs_fit(steps$model)
ols_plot_resid_qq(steps$model)
ols_plot_resid_fit(steps$model)

# Based on the Stepwise Regression, we decided to keep only the "Overall.Score","Weekly.Salary","Age","Defence",  
# "Weight","Ball.Skills","Goalkeeping",and "Physical" variables. 

#------------------------------------------------------------------------------------------------------------

# Now we will explore data transformation. 

hist(data$Goalkeeping)
# Goalkeeping values appear to be separated into two clusters. 
# The rationale behind this could be that high goalkeeping values only apply to goalkeepers. It would be helpful to
# divide that Goalkeeping values into two groups, and base on the plot, a good cutoff is 40.0 to create dummies.
# We will add an additional dummy variable for Goalkeeping.

data_train_copy$Goal_keep <- ifelse(data_train_copy$Goalkeeping > 40 ,1,0)

# Note that from previous distribution plots, 'Market.Value' and 'Weekly.Salary' 
# showed high skewness. We will consider to log-transform these variables. 

par(mfrow=c(1,2))
hist(data$Market.Value)
hist(log(data$Market.Value))

par(mfrow=c(1,2))
hist(data$Weekly.Salary)
hist(log(data$Weekly.Salary))


# Log transformation is helpful to mitigate skewness for these variables. 
# We will perform log transformation.
data_train_copy$Market.Value <- log(data_train$Market.Value)
data_train_copy$Weekly.Salary <- log(data_train$Weekly.Salary)
data_train_copy$Goalkeeping <- log(data_train$Goalkeeping)

# Now let's run the transformed model with selected features.
model2 <- lm(Market.Value ~ Overall.Score + Weekly.Salary + Age 
             + Defence + Weight + Ball.Skills + Physical + Goalkeeping
             + Goal_keep, data=data_train_copy)
summary(model2)

# Note both the R^2 value and adjusted R^2 value significantly improved


#####################################
##### Q3. Model Selection############
#####################################

# We have already performed variable transformation in question 2 and observed that doing log-transformation and creating dummy variables
# helped improve our model's accuracy. 

# Before doing lasso regression or ridge regression, we should standardize our data. 

# X_train only contains selected variables from stepwise
X_train <- as.data.frame(data_train_copy[,steps$predictors])

# Only scale continuous variables
X_train_scaled <- cbind(scale(X_train), data_train_copy$Goal_keep)
colnames(X_train_scaled) <- c('Overall.Score', 'Weekly.Salary', 'Age','Defence', 'Weight','Ball.Skills', 
                              'Goalkeeping', 'Physical', 'Goal_keep')
X_train_scaled <- as.matrix(X_train_scaled)


Y_train <- data_train_copy[['Market.Value']]
Y_train_scaled <- scale(Y_train)

# It's also useful to select models based on performance on test data. We now perform the same variable transformation on the test data.
data_test_copy$foot_dummy <- ifelse(data_test_copy$Preferred.Foot=='Right',1,0)
data_test_copy$Preferred.Foot <- NULL
data_test_copy$Goal_keep <- ifelse(data_test$Goalkeeping > 40 ,1,0)
data_test_copy$Market.Value <- log(data_test$Market.Value)
data_test_copy$Weekly.Salary <- log(data_test$Weekly.Salary)
data_test_copy$Goalkeeping <- log(data_test$Goalkeeping)

X_test <- as.data.frame(data_test_copy[,steps$predictors])

# Only scale continuous variables
X_test_scaled <- cbind(scale(X_test), data_test_copy$Goal_keep)
colnames(X_test_scaled) <- c('Overall.Score', 'Weekly.Salary', 'Age','Defence', 'Weight','Ball.Skills', 
                              'Goalkeeping', 'Physical', 'Goal_keep')
X_test_scaled <- as.matrix(X_test_scaled)

Y_test <- data_test_copy[['Market.Value']]
Y_test_scaled <- scale(Y_test)

# Create a evaluation function 

evaluation <- function(actual, fitted, n){
  sse <- sum((fitted - actual)^2)
  sst <- sum((actual - mean(actual))^2)
  r_square <- 1 - sse / sst
  rmse = sqrt(sse/n)
  print(paste0("R-squared: ", round(r_square,4)))
  print(paste0("RMSE: ", round(rmse,4)))
}

#-------------------------------------------------------------------------------------------------
### Lasso Regression ###

library(glmnet)

# Create different values for the regularization parameter lambda

lambdas <- 10^seq(-5, 5, length.out = 50)

##Use Cross Validation to select the optimal tuning parameters##

cv.lasso <- cv.glmnet(X_train_scaled, Y_train_scaled, alpha = 1, lambda = lambdas, standardize=FALSE, nfolds = 10)

lasso.optimal_lambda <- cv.lasso$lambda.min

# Best lambda
lasso.optimal_lambda

cv.lasso$glmnet.fit

par(mfrow=c(1,1))
plot(cv.lasso)

# Evaluate optimal Lasso model selected by CV
cv.lasso.best <- glmnet(X_train_scaled, Y_train_scaled, alpha = 1, lambda = lasso.optimal_lambda, standardize=TRUE)
# Scores on training data
Y_train_predicted <- predict.glmnet(cv.lasso.best, s=lasso.optimal_lambda, newx = X_train_scaled)
evaluation(Y_train_scaled, Y_train_predicted, nrow(X_train_scaled))
# Scores on test data
Y_test_predicted <- predict.glmnet(cv.lasso.best, s=lasso.optimal_lambda, newx = X_test_scaled)
evaluation(Y_test_scaled, Y_test_predicted, nrow(X_test_scaled))

##Use AIC to select the optimal tuning parameters##

AIC <- c()
# Loop through the candidate lambdas 
for (i in seq(lambdas)){
  lasso <- glmnet(X_train_scaled,Y_train_scaled, alpha = 1, lambda = lambdas[i], standardize=TRUE)
  AIC[i] <- deviance(lasso)+2*(lasso$df) 
}

plot(log(lambdas), AIC, col = 'red', type = 'l', ylab='AIC')

#Optimal lambda based on AIC
optimal_lambda_aic <- lambdas[which.min(AIC)]
optimal_lambda_aic

# Evaluate optimal Lasso model selected by AIC
aic.lasso.best <- glmnet(X_train_scaled,Y_train_scaled, alpha = 1, lambda = optimal_lambda_aic, standardize=TRUE)
Y_train_predicted <- predict.glmnet(aic.lasso.best, s=optimal_lambda_aic, newx = X_train_scaled)
evaluation(Y_train_scaled, Y_train_predicted, nrow(X_train_scaled))

Y_test_predicted <- predict.glmnet(aic.lasso.best, s=optimal_lambda_aic, newx = X_test_scaled)
evaluation(Y_test_scaled, Y_test_predicted, nrow(X_test_scaled))

#--------------------------------------------------------------------------------------------------
### Ridge Regression ###


# Create different values for the regularization parameter lambda

lambdas <- 10^seq(-5, 5, length.out = 50)

##Use Cross Validation to select the optimal tuning parameters##

cv.ridge <- cv.glmnet(X_train_scaled, Y_train_scaled, alpha = 0, lambda = lambdas, standardize=TRUE, nfolds = 10)

ridge.optimal_lambda <- cv.ridge$lambda.min

# Best lambda
ridge.optimal_lambda

cv.ridge$glmnet.fit

par(mfrow=c(1,1))
plot(cv.ridge)

# Evaluate optimal Ridge model selected by CV
cv.ridge.best <- glmnet(X_train_scaled, Y_train_scaled, alpha = 0, lambda = ridge.optimal_lambda, standardize=TRUE)
Y_train_predicted <- predict.glmnet(cv.ridge.best, s=ridge.optimal_lambda, newx = X_train_scaled)
evaluation(Y_train_scaled, Y_train_predicted, nrow(X_train_scaled))

Y_test_predicted <- predict.glmnet(cv.ridge.best, s=ridge.optimal_lambda, newx = X_test_scaled)
evaluation(Y_test_scaled, Y_test_predicted, nrow(X_test_scaled))

##Use AIC to select the optimal tuning parameters##

AIC2 <- c()
# Loop through the candidate lambdas 
for (i in seq(lambdas)){
  ridge <- glmnet(X_train_scaled,Y_train_scaled, alpha = 0, lambda = lambdas[i], standardize=TRUE)
  AIC2[i] <- deviance(ridge)+2*(ridge$df) 
}

plot(log(lambdas), AIC2, col = 'red', type = 'l', ylab='AIC')

#Optimal lambda based on AIC
optimal_lambda_aic2 <- lambdas[which.min(AIC2)]
optimal_lambda_aic2

# Evaluate optimal Ridge model selected by AIC
aic.ridge.best <- glmnet(X_train_scaled,Y_train_scaled, alpha = 0, lambda = optimal_lambda_aic2, standardize=TRUE)
Y_train_predicted <- predict.glmnet(aic.ridge.best, s=optimal_lambda_aic2, newx = X_train_scaled)
evaluation(Y_train_scaled, Y_train_predicted, nrow(X_train_scaled))

Y_test_predicted <- predict.glmnet(aic.ridge.best, s=optimal_lambda_aic2, newx = X_test_scaled)
evaluation(Y_test_scaled, Y_test_predicted, nrow(X_test_scaled))

#-------------------------------------------------------------------------------------------------

#FINAL OPTIMIZED MODEL: Ridge Regression selected by Cross Validation. Best lambda = 1e-05
#Since its R^2 is the highest while the RMSE is lowest. 

Optimal_Final_Model <- cv.ridge.best

#--------------------------------------------------------------------------------------------------



#####################################
##### Q4. Model Evaluation###########
#####################################

###Recap final model###
cv.ridge.best <- glmnet(X_train_scaled, Y_train_scaled, alpha = 0, lambda = ridge.optimal_lambda, standardize=TRUE)


###Train-Test Split###
# Already done in Question 2#

###Evaluation###
Y_train_predicted <- predict.glmnet(cv.ridge.best, s=ridge.optimal_lambda, newx = X_train_scaled)
evaluation(Y_train_scaled, Y_train_predicted, nrow(X_train_scaled))

Y_test_predicted <- predict.glmnet(cv.ridge.best, s=ridge.optimal_lambda, newx = X_test_scaled)
evaluation(Y_test_scaled, Y_test_predicted, nrow(X_test_scaled))
#Our model performs well on both training and test data. 


#Residual vs. fit Plot
fitted <- predict.glmnet(cv.ridge.best, s=ridge.optimal_lambda, newx = X_train_scaled)
residuals <- Y_train_scaled - fitted
plot(fitted, residuals, xlab = "Fits", ylab = "Residuals_squared")
#From the residual vs. fit plot, we can see that most residuals center around 0, except a few outliers. This suggests
#that the linear relationship assumption holds. 
#Residuals roughly form a band around the 0 line, which suggests that the variances of the error terms are equal. 
qqnorm(residuals)
#From the Q-Q plot, we can see the residuals fairly follows a normal distribution. 


