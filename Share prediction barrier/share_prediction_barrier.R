rm(list=ls())
library(readxl)
library(dplyr)
library(lubridate)
library(quantmod)
library(xgboost)
library(randomForest)
library(tseries)
library(caret)
library(data.table)
#Set parameters
start <- as.Date("1999-05-01")
end <- as.Date("2019-05-01")
barrier = 0.5
ticker = "GE"
share_price = 9.385647

aux =getSymbols(ticker,auto.assign = FALSE, from =start, to = end)
aux2 = as.data.frame(aux)
aux2$date=rownames(aux2)
sapply(aux2,class)
aux2$date=ymd(aux2$date)
#Address missing values----
library(imputeTS)
#Make a sequense of dates from Monday to Friday
dd=data.frame(date=seq(min(aux2$date), max(aux2$date), "days"))
dd$day=wday(dd$date,week_start = 1)
#dd=left_join(dd,aux2[,6:7],by="date")
dd2=left_join(dd,aux2[,1:7], by="date")
colSums(is.na(dd2)) # 2272
# Interpolate missing values
dd2$open=na_interpolation(dd2$GE.Open,option="linear")
dd2$high=na_interpolation(dd2$GE.High,option="linear")
dd2$low=na_interpolation(dd2$GE.Low,option="linear")
dd2$close=na_interpolation(dd2$GE.Close,option="linear")
dd2$volume=na_interpolation(dd2$GE.Volume,option="linear")
dd2$adj=na_interpolation(dd2$GE.Adjusted,option="linear")
colSums(is.na(dd2))
#Exponentially smothing
#View prices
library(forecast)
windows()
plot(dd2$date,dd2$adj,type="l",col="blue",lwd=2,las=2,xlab="",ylab="")
#Add a threshold if share falls below the barrier
#barrier for the share percent of price of the issuing date
barrier_share_price = share_price*barrier
dd2$target = ifelse(dd2$adj>barrier_share_price,0,1)
#Use log returns
#Open price
dd2$open_l1=dplyr::lag(dd2$open,n=1)
dd2$log_rets_open=log(dd2$open/dd2$open_l1)
#High price
dd2$high_l1=dplyr::lag(dd2$high,n=1)
dd2$log_rets_high=log(dd2$high/dd2$high_l1)
#Low price
dd2$low_l1=dplyr::lag(dd2$low,n=1)
dd2$log_rets_low=log(dd2$low/dd2$low_l1)
#Close price
dd2$close_l1=dplyr::lag(dd2$close,n=1)
dd2$log_rets_close=log(dd2$close/dd2$close_l1)
#Volume
#Adjusted price
dd2$adj_l1=dplyr::lag(dd2$adj,n=1)
dd2$log_rets_adj=log(dd2$adj/dd2$adj_l1)

#remove unnecessary columns
dd2$day = NULL
dd2$GE.Open = NULL
dd2$GE.High = NULL
dd2$GE.Low = NULL
dd2$GE.Close = NULL
dd2$GE.Volume = NULL
dd2$GE.Adjusted = NULL
dd2$open_l1 = NULL
dd2$high_l1 = NULL
dd2$low_l1 = NULL
dd2$close_l1 = NULL
dd2$adj_l1 = NULL
dd2$open = NULL
dd2$high = NULL
dd2$low = NULL
dd2$close = NULL
dd2$adj = NULL
#remove nas
dd2=na.omit(dd2)

#Making and ADF Test for statinarity
adf.test(dd2$log_rets_adj)


#Making additional variables----
#1.Money flow index
dd2$mfi = MFI(dd2[,c("log_rets_open","log_rets_low","log_rets_close")], dd2[,"volume"])
#2.Relative Strengh Index
price=dd2$log_rets_adj
dd2$rsi= RSI(price, n = 14)
#3.MACD Oscillator
dd2$macd = MACD( price, 12, 14, 9, maType="EMA" )
#4.On Balance Volume
#dd2$obv = OBV(price,dd2$volume)
#5.

dd2=na.omit(dd2)
#table
target= dd2$target
mfi= dd2$mfi
rsi=dd2$rsi
macd=dd2$macd 
var1=cbind(mfi,rsi,macd)


#scale the variables
var1 = data.frame(scale(var1))
#table
t= cbind(target,var1)

#Making a model
#XGBOOST----
set.seed(123)
ind = sample(2, nrow(t),replace = T, prob = c(0.7, 0.3))
train = t[ind==1,]
test = t[ind==2,]
x.train = train[,-1] 
x.test = test[,-1]
y.train = as.matrix(train[1])
y.test =  as.matrix(test[1])
y.test.not.matrix = test[1]
#Prepare the data
dtrain = xgb.DMatrix(as.matrix(sapply(x.train, as.numeric)), label = y.train)
dtest= xgb.DMatrix(as.matrix(sapply(x.test, as.numeric)), label=y.test)
# Fit the model
watchlist=list(train=dtrain, test=dtest)
eq=xgb.train(data=dtrain, max_depth=2, eta=1, nthread = 2, nrounds=10, watchlist=watchlist, objective = "binary:logistic")
# Make predictions
label=getinfo(dtest, "label")
eq_pred=predict(eq, dtest)
summary(eq_pred)
# Determine the size of the prediction vector
print(length(eq_pred))
# Limit display of predictions to one year 250(working days)
print(head(eq_pred,250))
prediction = as.numeric(eq_pred > 0.5)
prediction = as.data.frame(prediction)
#Chance of hitting the barrier
table(prediction)

#Random Forest
#Random Forest----
eq2 = randomForest(target ~ ., data=train, proximity=FALSE,importance = FALSE,
                    ntree=500,mtry=4, do.trace=FALSE)

eq_2pred = predict(eq2, newdata=test)

summary(eq2)

print(eq_2pred)
