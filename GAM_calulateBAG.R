
library(mgcv)
#read data
data<-read.csv('/roi_HMD_data_ICV.csv', header=TRUE)
data$Sex<-as.factor(data$Sex)

HC_data<-data[data$Group==0,]
psy_data<-data[data$Group>0,]


#make a list to store results
storage<-list()
for (i in names(HC_data)[44:199]){storage[[i]]<-predict(gam(get(i) ~ s(Age)+Sex+s(Age,by=Sex), data=HC_data), newdata=data)}
sink(file="GAM_all_pred.txt")
storage

#pred_data using HC
pred_DF <- as.data.frame(storage)
write.csv(pred_DF, "pred_data.csv")
#bag data
bag_DF<- data[44:199]-pred_DF
write.csv(bag_DF, "bag_data.csv")
