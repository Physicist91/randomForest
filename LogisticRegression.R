###########################################
### Credit Card Default data
###########################################
library("ISLR")
?Default
head(Default)
summary(Default)

## linear regression: default ~ balance
Default$default0 <- as.numeric(Default$default) - 1
lm0 <- lm(default0 ~ balance, data=Default)
summary(lm0)
plot(Default$balance, Default$default0)
abline(lm0, col="red", lwd=2)


## Logistic regression 1: default ~ balance
glm1 <- glm(default ~ balance, data=Default, family=binomial(link="logit"))
summary(glm1)

str(glm1)
points(Default$balance, glm1$fitted.values, col="blue")

predict(glm1, newdata=data.frame(balance=c(1000, 2000)), type="response")


## Logistic regression 2: default ~ student
glm2 <- glm(default ~ student, data=Default, family=binomial(link="logit"))
summary(glm2)

plot(Default$student, glm2$fitted.values)


## Logistic regression 3: default ~ .
glm3 <- glm(default ~ balance + income + student, data=Default, family=binomial(link="logit"))
summary(glm3)


## model selectin by AIC
glm.best <- stepAIC(glm3, direction="both")
summary(glm.best)


## update()
glm4 <- update(glm3, . ~ . - income)
summary(glm4)


confint(glm4) # 95% CI for the coefficients
exp(coef(glm4)) # exponentiated coefficients
exp(confint(glm4)) # 95% CI for exponentiated coefficients
fitted(glm4, type="response") # fitted values
residuals(glm4, type="deviance") # residuals

# more statistics
par(mfrow=c(2,2))
plot(glm4)
par(mfrow=c(1,1))


## categorical prediction
## use fixed cutoff = 0.5
Default$glm4.prob <- predict(glm4, type="response")
Default$glm4.pred <- Default$glm4.prob > 0.5
confusion.mat <- table(Default$glm4.pred, Default$default)
confusion.mat
TP <- confusion.mat[2, 2]
TN <- confusion.mat[1, 1]
FP <- confusion.mat[2, 1]
FN <- confusion.mat[1, 2]
recall <- TP / (TP + FN)
specificity <- TN / (FP + TN)
precision <- TP / (TP + FP)
accuracy <- (TP + TN) / nrow(Default)


## plot measures against the cutoff
library("ROCR")
glm4.pred <- prediction(Default$glm4.prob, Default$default)

?performance

# misclassificatin vs. fpr vs. fnr
glm4.fpr = performance(glm4.pred, measure="fpr")
glm4.fnr = performance(glm4.pred, measure="fnr")
glm4.err = performance(glm4.pred, measure="err")
# plot in one figure
plot(glm4.fpr, col="red", ylab="")
plot(glm4.fnr, col="blue", add=TRUE)
plot(glm4.err, col="black", add=TRUE)
legend(x=0.7, y=0.5, legend=c("Error Rate", "False Positive Rate", "False Negative Rate"), lty=c(1, 1, 1), lwd=c(2, 2, 2), col=c("black", "red", "blue"))

# accuracy vs. cutoff
glm4.acc = performance(glm4.pred, measure="acc")
plot(glm4.acc)
# precision vs. cutoff
glm4.prec = performance(glm4.pred, measure="prec")
plot(glm4.prec)

# ROC plot
glm4.ROC <- performance(glm4.pred, measure = "tpr", x.measure = "fpr")
plot(glm4.ROC)
abline(a=0, b=1, lty=2) # diagonal line
# AUC
as.numeric(performance(glm4.pred, "auc")@y.values)

# Precision-Recall plot
glm4.PR <- performance(glm4.pred, measure = "prec", x.measure = "rec")
plot(glm4.PR)

# Sensitivity-Specificity plot
glm4.SS <- performance(glm4.pred, measure = "sens", x.measure = "spec")
plot(glm4.SS)

# Lift chart
glm4.lift <- performance(glm4.pred, measure = "lift", x.measure = "rpp")
plot(glm4.lift)




## Probit regression: default ~ .
glm.probit <- glm(default ~ balance + student, data=Default, family=binomial(link="probit"))
summary(glm.probit)



