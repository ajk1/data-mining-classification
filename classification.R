# libraries
library(e1071)
library(class)
library(ggplot2)
library(reshape2)

# utility function for import from csv file
import.csv <- function(filename) {
  return(read.csv(filename, sep = ",", header = TRUE))
}

# import wine dataframe
wine <- import.csv("wine.csv")

# assign "low" to 0 and "high" to 1
type.num <- c()
for (row in 1:nrow(wine)) {
  if (wine[row, ncol(wine)] == "low") {
    type.num <- c(type.num, 0)
  }
  else if (wine[row, ncol(wine)] == "high") {
    type.num <- c(type.num, 1)
  }
  else {
    type.num <- c(type.num, NA)
  }
}
wine[,4] <- type.num

# remove rows with NA values (none for this set)
wine <- wine[which(complete.cases(wine)),]

# each model assumes the last column of train holds the output dimension

# logistic regression
get_pred_logreg <- function(train, test) {
  m <- glm(as.formula(paste(colnames(train)[ncol(train)], "~ .")),
           family=binomial(link="logit"), data = train)
  pred <- as.vector(predict(m, test))
  pred <- exp(pred)/(1+exp(pred)) # maps predictions to values between 0 and 1
  pred <- data.frame(pred, test[,ncol(test)])
  colnames(pred) <- c("prediction", "true")
  return(pred)
}

# support vector model
get_pred_svm <- function(train, test) {
  m <- svm(as.formula(paste(colnames(train)[ncol(train)], "~ .")), data = train)
  pred <- data.frame(as.vector(predict(m,test)), test[,ncol(test)])
  colnames(pred) <- c("prediction", "true")
  return(pred)
}

# naive bayes
get_pred_nb <- function(train, test) {
  m <- naiveBayes(train[,-ncol(train)], train[,ncol(train)])
  # predict with type "raw" will return two columns, first with pr("low") 
  # and second with pr("high") - we want the second column as our prediction
  pred <- data.frame(predict(m, test[,-ncol(test)], type="raw")[,2], test[,ncol(test)])
  colnames(pred) <- c("prediction", "true")
  return(pred)
}

# k-nearest neighbor
get_pred_knn <- function(train, test, k) {
  m <- knn(train[,1:ncol(train)-1], test[,1:ncol(test)-1], 
           cl = train[,ncol(train)], k = k)
  m <- ifelse(m=="1", 1, 0) # converts m from factor to numeric
  pred <- data.frame(m, test[,ncol(test)])
  colnames(pred) <- c("prediction", "true")
  return(pred)
}

# k-fold cross-validation
do_cv <- function(df, num_folds, model_name){
  # randomly shuffle rows
  rand = sample.int(nrow(df)) #random permutation of 1:nrow(wine)
  df1 <- df
  for(row in 1:nrow(df)){
    df[row,] <- df1[rand[row],]
  }
  
  # create vector of start indices for each partition of wine
  partition = c()
  start = 1
  for(i in 1:(num_folds+1)){
    partition = c(partition, start)
    start = as.integer(nrow(wine)*(i/num_folds))+1
  }
  
  # call function and calculate performance metrics for each data fold
  
  metrics <- data.frame()
  
  if (substr(model_name, nchar(model_name)-1, nchar(model_name)) == "nn") {
    for(i in 1:num_folds){
      test <- wine[ partition[i]:(partition[i+1]-1), ] 
      train <- wine[ -c(partition[i]:(partition[i+1]-1)), ]
      # call corresponding model function and get metrics
      pred <- get_pred_knn(train, test, as.numeric(substr(model_name,1,nchar(model_name)-2)))
      metrics <- rbind(metrics, get_metrics(pred))
    }
  }
  else {
    for(i in 1:num_folds){
      test <- wine[ partition[i]:(partition[i+1]-1), ] 
      train <- wine[ -c(partition[i]:(partition[i+1]-1)), ]
      pred <- do.call(paste("get_pred_", model_name, sep=""), list(train, test))
      roc <- get_roc(pred)
#       a <- ggplot(data=roc, aes(x=fpr, y=tpr)) + geom_point() +
#         ggtitle(paste(model_name, " roc", i, sep=""))
#       print(a)
      cutoff <- ifelse(model_name=="nb", .6, .5)
      metrics <- rbind(metrics, cbind(get_metrics(pred, cutoff), get_auc(roc)))
    }
  }
  return(sapply(metrics, mean))
}

# first column of pred contains predicted values, the second represents true values (0 or 1)
get_metrics <- function(pred, cutoff=0.5) {
  tp = length(which(pred[,1]>cutoff & pred[,2]==1))
  fp = length(which(pred[,1]>cutoff & pred[,2]==0))
  pos = length(which(pred[,2]==1))
  neg = length(which(pred[,2]==0))
  tpr <- tp/pos
  fpr <- fp/neg
  acc <- (pos*tpr + neg*(1-fpr))/(pos+neg)
  precision <- tp/(tp+fp)
  recall <- tpr
  return(data.frame(tpr, fpr, acc, precision, recall))
}

# gets x and y values for roc scatterplot
get_roc <- function(pred) {
  pred = pred[order(pred[,1],decreasing = TRUE),]
  x <- c(0)
  y <- c(0)
  for (i in 1:nrow(pred)) {
    x[i+1] <- x[i] + ifelse(pred[i,2]==1, 0, 1)
    y[i+1] <- y[i] + ifelse(pred[i,2]==1, 1, 0)
  }
  fpr <- x/length(which(pred[,2]==0))
  tpr <- y/length(which(pred[,2]==1))
  roc <- data.frame(fpr,tpr)
  return(roc)
}

# gets area under curve of given roc data frame
get_auc <- function(roc) {
  prev <- 0
  auc <- 0
  for (i in 1:nrow(roc)) {
    if (roc[i,1] != prev) {
      auc <- auc + roc[i,2]
      prev <- roc[i,1]
    }
  }
  return(auc/length(unique(roc[,1])))
}

# test KNN with varying values of k
test_knn <- function(num_folds) {
  metrics <- data.frame()
  for (k in 1:30) {
    metrics <- rbind(metrics, c(k, do_cv(wine, num_folds, paste(k,"nn", sep=""))))
  }
  colnames(metrics) <- c("k", "tpr", "fpr", "accuracy", "precision", "recall")
  print(metrics)
  dm <- melt(metrics[,c(1,4:6)], id.vars="k")
  a <- ggplot(data=dm,aes(x=k,y=value,colour=variable)) +
    scale_colour_manual(values=c("red","blue", "black")) +
    geom_point() + xlab("Number of nearest neighbors") + ylab("Measure") + 
    ggtitle(paste("Performance metrics for ", num_folds, "-fold CV of K-NN for varying K", sep=""))
  print(a)
}

cat("\nPerformance metrics for K-NN:\n")
cat("24 fold CV\n")
test_knn(num_folds=24)

# test parametric models
test_parametric <- function(num_folds) {
  metrics <- data.frame()
  metrics <- rbind(metrics, do_cv(wine, num_folds, "logreg"))
  metrics <- rbind(metrics, do_cv(wine, num_folds, "svm"))
  metrics <- rbind(metrics, do_cv(wine, num_folds, "nb"))
  metrics <- cbind(c("logreg","svm","nb"), metrics)
  colnames(metrics) <- c("model", "tpr", "fpr", "accuracy", "precision", "recall", "auc(roc)")
  print(metrics)
}

cat("\nPerformance metrics for parametric models:\n")
cat("6 fold CV\n")
test_parametric(num_folds=6)
cat("12 fold CV\n")
test_parametric(num_folds=12)
cat("18 fold CV\n")
test_parametric(num_folds=18)
cat("24 fold CV\n")
test_parametric(num_folds=24)

cat("\nDefault classifier accuracy:\n")
print(length(which(wine[,ncol(wine)]==1))/nrow(wine))
