# Reference:
# - https://github.com/trestletech/plumber
# - https://github.com/Knowru/plumber_example/blob/master/deploy_ml_credit_model.R

library(rpart)
library(jsonlite)
load("decision_tree_for_german_credit_data.RData")

#* @get /
health.check <- function() {
    return("ok")
}

#* @post /predict
predict.default.rate <- function(
d=list(Status.of.existing.checking.account
  , Duration.in.month
  , Credit.history
  , Savings.account.bonds)
) {
  data <- d
#  data <- c(
#    Status.of.existing.checking.account=Status.of.existing.checking.account
#    , Duration.in.month=Duration.in.month
#    , Credit.history=Credit.history
#    , Savings.account.bonds=Savings.account.bonds
#  )
  Status.of.existing.checking.account = c("A14", "A15", "A11")
  Duration.in.month = c(12, 11, 10)
  Credit.history = c("A34", "A33", "A31")
  Savings.account.bonds = c("A64", "A61", "A60")
  df = data.frame(Status.of.existing.checking.account, Duration.in.month, Credit.history, Savings.account.bonds)
  df.list <- as.list(as.data.frame(t(df)))
  prediction <- predict(decision.tree, df.list)
  return(list(default.probability=unbox(prediction[1, 1])))
}
