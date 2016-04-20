### Regularization in Regression
library("glmnet")
library("ISLR")

names(Hitters)
dim(Hitters)

# the Salary column has NA's
sum(is.na(Hitters$Salary))
# remove NA's in Salary
Hitters <- na.omit(Hitters)
sum(is.na(Hitters))
dim(Hitters)
summary(Hitters)



# test the model.matrix function
m <- model.matrix(Salary ~ Hits + HmRun, Hitters)
head(m)
str(m)


# construct x and y matrix for glmnet()
x <- model.matrix(Salary ~ ., Hitters)[,-1]
y <- Hitters$Salary


## Ridge Regression

# first define a series of values as potential lambda
lambdas <- 10 ^ seq(6,-1,length=100)
plot(lambdas)

# run Ridge Regression using glmnet() (alpha=0 means Ridge Regression)
library("glmnet")
ridge.mod <- glmnet(x, y, alpha=0, lambda=lambdas)

# check the fitted model
str(ridge.mod)

# plot the fitted model
plot(ridge.mod, xvar="lambda", label=TRUE)

# check the coefficient output
dim(coef(ridge.mod))  # 20 coefficients for the 19 variables, for each lambda
coef(ridge.mod)

# for example, the 50-th lambda
ridge.mod$lambda[50]
coef(ridge.mod)[,50]
# the penalty term for the 50-th lambda
sqrt(sum(coef(ridge.mod)[-1,50]^2))

# compare with the 60-th lambda
ridge.mod$lambda[60]
coef(ridge.mod)[,60]
# the penalty term for the 60-th lambda
sqrt(sum(coef(ridge.mod)[-1,60]^2))

# can also specify a lambda (s=50) and "predict" the coefficients
predict(ridge.mod, s=50, type="coefficients")





## use CV to find optimal lambda
# CV
set.seed(1)
ridge.cv <- cv.glmnet(x, y, alpha=0, lambda=lambdas)
plot(ridge.cv)

# optimal lambda
ridge.lam <- ridge.cv$lambda.min
log(ridge.lam)
points(log(ridge.lam), min(ridge.cv$cvm), cex=3)

# plot optimal lambda
plot(ridge.mod, xvar="lambda", label = TRUE)
abline(v=log(ridge.lam), lty=2)

# check the coefficients at the optimal lambda
coef(ridge.cv, s="lambda.min")
# alternatively
predict(ridge.cv, type="coefficient", s="lambda.min")

# prediciton using optimal lambda
# predict(ridge.mod, s=ridge.lam, newx=x.test)




## The Lasso
lasso.mod <- glmnet(x, y, alpha=1, lambda=lambdas)
plot(lasso.mod, xvar="lambda", label=TRUE)

# CV
set.seed(1)
lasso.cv <- cv.glmnet(x, y, alpha=1, lambda=lambdas)
plot(lasso.cv)

# optimal lambda
lasso.lam <- lasso.cv$lambda.min
log(lasso.lam)
points(log(lasso.lam), min(lasso.cv$cvm), cex=3)

# plot optimal lambda
plot(lasso.mod, xvar="lambda", label = TRUE)
abline(v=log(lasso.lam), lty=2)

# check the coefficients at the optimal lambda
coef(lasso.cv, s="lambda.min")
# alternatively
predict(lasso.mod, type="coefficients", s=lasso.lam)

# prediciton using optimal lambda
# predict(lasso.mod, s=lasso.lam, newx=x.test)





## The Elastic Net with a given alpha
en.mod <- glmnet(x, y, alpha=0.5, lambda=lambdas)
plot(en.mod, xvar="lambda", label=TRUE)

# CV
set.seed(1)
en.cv <- cv.glmnet(x, y, alpha=0.5, lambda=lambdas)
plot(en.cv)

# optimal lambda
en.lam <- en.cv$lambda.min
log(en.lam)
points(log(en.lam), min(en.cv$cvm), cex=3)

# plot optimal lambda
plot(en.mod, xvar="lambda", label = TRUE)
abline(v=log(en.lam), lty=2)

# check the coefficients at the optimal lambda
coef(en.cv, s="lambda.min")
# alternatively
predict(en.mod, type="coefficients", s=en.lam)





## The Elastic Net with optimal alpha
lambdas <- exp(seq(-2, 6, 0.01))
set.seed(258)
folds <- sample(1:10, size=length(y), replace=TRUE)
cv1 <- cv.glmnet(x, y, foldid=folds, alpha=1, lambda=lambdas)
cv.75 <-cv.glmnet(x, y, foldid=folds, alpha=.75, lambda=lambdas)
cv.5 <-cv.glmnet(x, y, foldid=folds, alpha=.5, lambda=lambdas)
cv.25 <-cv.glmnet(x, y, foldid=folds, alpha=.25, lambda=lambdas)
cv0 <- cv.glmnet(x, y, foldid=folds, alpha=0, lambda=lambdas)

min(cv1$cvm)
min(cv.75$cvm)
min(cv.5$cvm)
min(cv.25$cvm)
min(cv0$cvm)

plot(log(cv1$lambda), cv1$cvm, col="red", xlab="log(Lambda)", ylab=cv1$name, type="l", ylim=c(115000, 140000))
lines(log(cv.75$lambda), cv.75$cvm, col="grey")
lines(log(cv.5$lambda), cv.5$cvm, col="grey")
lines(log(cv.25$lambda), cv.25$cvm, col="grey")
lines(log(cv0$lambda), cv0$cvm, col="blue")


