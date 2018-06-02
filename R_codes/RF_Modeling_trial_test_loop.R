RF_Modeling_test <- function(filenum){
# Add Random Forest Library
library(randomForest)
library(caret)
library(mnormt)
library(rpart)
library(clusterGeneration)
library(help=randomForest)
help(randomForest)
library(dplyr)
library(kernlab)
library(e1071)
library(corrplot)
tree=30
csv=5
ts_percent=90
fst_column="Id"
endpoint="Yes"
cor_val=0.85
imp_val=10

#Path to the .csv file (smiles + Descriptors from PCAT)
dir="/Users/dptrujillo/Dropbox/RF_Modelling_Test/Trial_Data/"	
dir2="/Users/dptrujillo/Dropbox/RF_Modelling_Test/Results/"
#input filename (.csv file from Ayana)			
Filename="File1"
Filename_2="File2"
Filename_3="File3"
#Output="Results"																
ext=".csv"




print (paste("========================================================================"))
print (paste("       Model 1:  All Descriptors after removing zero variance &         "))
print (paste("       highly correlated                                                "))
print (paste("========================================================================="))


input=paste(dir,Filename,ext,sep="") 

# Reading input file
mydata <- data.frame(read.csv(file=input))
print (paste("Dimensions of data file"))
print (paste("Number of Rows:    ", nrow(mydata)))
print (paste("Number of Columns: ", ncol(mydata)))

mydataset <- mydata[ ,!(colnames(mydata) %in% c(fst_column))]
mydataDescr <- data.frame(mydataset[ ,!(colnames(mydataset) %in% c(endpoint))])

print (paste("            ****              "))
print (paste("Dimensions of Descriptors"))
print (paste("Number of Rows:    ", nrow(mydataDescr)))
print (paste("Number of Columns: ", ncol(mydataDescr)))

print (paste("======================================"))
print (paste("       Removing Zero Variance         "))
print (paste("======================================"))



nzv <- nearZeroVar(mydataDescr)
mydataDescr <- mydataDescr[,-nzv]

print (paste("Dimensions of Descriptors"))
print (paste("Number of Rows:    ", nrow(mydataDescr)))
print (paste("Number of Columns: ", ncol(mydataDescr)))
print (paste("            ****              "))

print (paste("==================================================="))
print (paste("  Removing Highly Correlated: ", cor_val*100,"%"))
print (paste("==================================================="))
mydataDescr <- data.matrix(mydataDescr, rownames.force = NA)
mydataDescr[is.na(mydataDescr)] <- 0
print(paste("Here"))
tmp <- cor(mydataDescr, use="pairwise.complete.obs")
print(paste("Here1"))
tmp[!lower.tri(tmp)] <- 0
mydataDescr <- mydataDescr[,!apply(tmp,2,function(x) any(x > cor_val))]
print (paste("Dimensions of Descriptors"))
print (paste("Number of Rows:    ", nrow(mydataDescr)))
print (paste("Number of Columns: ", ncol(mydataDescr)))
print (paste("            ****              "))


mydatanew <- data.frame(mydataDescr)
mydatanew <- cbind(fst_column = 0, mydatanew)

mydatanew$Id<- mydata[,c(fst_column)]
mydatanew[endpoint] <- NA
mydatanew$Yes <- mydata[,c(endpoint)] 
mydatanew <- data.frame(mydatanew)

print (paste("======================================"))
print (paste("       Training and Test Sets         "))
print (paste("======================================"))

ts_size = floor(ts_percent*nrow(mydata)/100)
testSetSize = nrow(mydata)- ts_size

print (paste("The training set contains: ", ts_size, "molecules"))
print (paste("The test set contains: ", testSetSize, "molecules"))

randomset<-sample(nrow(mydatanew),ts_size)
trainingset <- mydatanew[randomset, ]
testset <- mydatanew[-randomset, ]
trainingset <- trainingset[ ,!(colnames(trainingset) %in% c(fst_column))]
testset<- testset[ ,!(colnames(testset) %in% c("Id"))]


observed_train<-data.frame(trainingset$Yes)
Id_train<-data.frame(mydatanew$Id)

observed_test<-data.frame(testset$Yes)
Id_test<-data.frame(mydatanew$Id)

#testfile=paste(dir2,'test',ext,sep="") 
#for (i in 0:length(trainingset[,1])){
#	for (j in 0:length(trainingset[1,])){
#		print(cat(trainingset[i,j],sep=" , ","\t",file=testfile,append=TRUE))
#	}
#}

print (paste("======================================"))
print (paste("       Id and Training Sets         "))
print (paste("======================================"))
train_out1=paste("Training_Results1_",tree,"_",nrow(mydatanew),"_TrainingSet_",ts_size,"_",filenum)
train_filename=paste(dir2,train_out1,ext,sep="") 
print (paste("written to:",train_filename))
print(cat("Id,Observed","\n",file=train_filename))
for (index in 1:length(trainingset[,1])){
	print(cat(Id_train[index,1],observed_train[index,1],sep=" , ","\n",file=train_filename,append=TRUE))
}

print (paste("======================================"))
print (paste("       Id and Test Sets         "))
print (paste("======================================"))
test_out1=paste("Test_Results1_",tree,"_",nrow(mydatanew),"_TestSet_",ts_size,"_",filenum)
test_filename=paste(dir2,test_out1,ext,sep="") 
print (paste("written to:",test_filename))
print(cat("Id,Observed","\n",file=test_filename))
for (index in 1:length(testset[,1])){
	print(cat(Id_test[index,1],observed_test[index,1],sep=" , ","\n",file=test_filename,append=TRUE))
}

#Probability plot (QQ plot) to see if the data is normally distributed
#qqnorm(trainingset$Yes, main = "Probability Plot")
print(paste("Here3"))
#qqline(trainingset$Yes)
print(paste("Here4"))

#In case the data is not normaly distributed Hypothesis test to complement our graphical analysis
#shapiro.test
#ad.test

#Training and test Descriptors
TrainingDescr <- data.frame(trainingset[ ,!(colnames(trainingset) %in% c(endpoint))])
TestDescr <- testset[ ,!(colnames(testset) %in% c(endpoint))]

print (names(TrainingDescr))
## Creating The Model
training_rmse_error = 0
validation_rmse_error  = 0
training_mae_error = 0
validation_mae_error  = 0


print (trainingset$Yes)
model <-randomForest(TrainingDescr,trainingset$Yes, mtry= csv, ntree=tree, keep.forest=TRUE, importance=TRUE, prox = TRUE)
print (summary(model$rsq))
plot(model)

#print (paste(test,test2))
training_rmse_error <- training_rmse_error + rmse(trainingset$Yes, predict(model, trainingset))
validation_rmse_error <- validation_rmse_error + rmse(testset$Yes, predict(model, testset))
training_mae_error <- training_mae_error + mae(trainingset$Yes, predict(model, trainingset))
validation_mae_error <- validation_mae_error + mae(testset$Yes, predict(model, testset))

out1=paste("Results1_",tree,"_",nrow(mydatanew),"_TrainingSet_",ts_size,"_TestSet_",testSetSize,"_",filenum)
output_filename=paste(dir2,out1,ext,sep="") 
print (output_filename)

print(paste("RMSE training set: ", training_rmse_error))
print(paste("RMSE test set:     ", validation_rmse_error))
print(paste("MAE training set:  ", training_mae_error))
print(paste("MAE test set:      ", validation_mae_error))
 
#print(importance(model, type = 1))
print(model)
predicted <- data.frame(predict(model, testset))
observed<-data.frame(testset$Yes)

print(cat(",Observed,Predicted","\n",file=output_filename))
for (index in 1:length(predicted[,1])){print(cat(index,observed[index,1],predicted[index,1],sep=" , ","\n",file=output_filename,append=TRUE))}

#out1=paste("Resultstrain_",tree,"_",nrow(mydatanew),"_TrainingSet_",ts_size,"_TestSet_",testSetSize)
#output_filename=paste(dir2,out1,ext,sep="") 
#print (output_filename)

#print(importance(model, type = 1))
#print(model)
#predicted <- data.frame(predict(model, trainingset))
#observed<-data.frame(trainingset$Yes)

#print(cat(",Observed,Predicted","\n",file=output_filename))
#for (index in 1:length(predicted[,1])){print(cat(index,observed[index,1],predicted[index,1],sep=" , ","\n",file=output_filename,append=TRUE))}


print (paste("============================================================="))
print (paste("       Model 2:  with Molecular & important Descriptors          "))
print (paste("============================================================="))
print (paste ("The Descriptors by order of importance are in the following file:"))


impDescr=paste("ImportantDescr_",tree,"_",nrow(mydatanew),"_TrainingSet_",ts_size,"_TestSet_",testSetSize)
output_filename=paste(dir,impDescr,ext,sep="")
print (paste(output_filename))



df <- importance(model, scale= F) 
write.table(df[-row(df)[df == 0],],file = output_filename, sep = ",", append = T, row.names = TRUE,col.names = F)
ImpVar <- data.frame(read.csv(file=output_filename))
names(ImpVar) <- c("Descr","imp1","imp2")


ImpVar <- ImpVar[order(ImpVar$imp2),]
ts= paste(ImpVar[1:imp_val,1])
data1 <- data.frame(mydata[,c(fst_column,"Chirality","CLOGP","NOCNT","HBD","MW","PSA","cPFLOGD","HBA","ROTB","Accptdonr","Boiling","Melting","Polarizability","Dielectric","Viscosity","DipoleM","Density","Surface","solvCLOGP","solvNOCNT","solvHBD","solvMW","solvPSA","solvHBA","solvROTB")])
print(paste("impval", imp_val))
data2 <- data.frame(mydatanew[,c(ts)])
data2[endpoint] <- NA
data2$Yes <- mydata[,c(endpoint)]
mydata_2 <- data.frame(cbind(data1,data2))



randomset<-sample(nrow(mydata_2),ts_size)
trainingset <- mydata_2[randomset, ]
testset <- mydata_2[-randomset, ]
trainingset <- trainingset[ ,!(colnames(trainingset) %in% c(fst_column))]
testset<- testset[ ,!(colnames(testset) %in% c(fst_column))]
observed_train<-data.frame(trainingset$Yes)
Id_train<-data.frame(mydatanew$Id)



#Probability plot (QQ plot) to see if the data is normally distributed
qqnorm(trainingset$Yes, main = "Probability Plot_2")
qqline(trainingset$Yes)


TrainingDescr <- data.frame(trainingset[ ,!(colnames(trainingset) %in% c(endpoint))])
TestDescr <- testset[ ,!(colnames(testset) %in% c(endpoint))]


training_rmse_error = 0
validation_rmse_error  = 0
training_mae_error = 0
validation_mae_error  = 0

model_2 <-randomForest(TrainingDescr,trainingset$Yes, mtry= csv, ntree=tree, keep.forest=TRUE, importance=TRUE, prox = TRUE)

#print (paste(test,test2))
training_rmse_error <- training_rmse_error + rmse(trainingset$Yes, predict(model_2, trainingset))
validation_rmse_error <- validation_rmse_error + rmse(testset$Yes, predict(model_2, testset))
training_mae_error <- training_mae_error + mae(trainingset$Yes, predict(model_2, trainingset))
validation_mae_error <- validation_mae_error + mae(testset$Yes, predict(model_2, testset))



out2=paste("Results2_",tree,"_",nrow(mydata_2),"_TrainingSet_",ts_size,"_TestSet_",testSetSize,"_",filenum)
output_filename=paste(dir2,out2,ext,sep="") 
print (output_filename)

print(paste("RMSE training set: ", training_rmse_error))
print(paste("RMSE test set:     ", validation_rmse_error))
print(paste("MAE training set:  ", training_mae_error))
print(paste("MAE test set:      ", validation_mae_error))
 
#print(importance(model, type = 1))
print(model_2)
#print (paste("R^2: ", summary(model)$rsq))
# Using Model to predict
predicted <- data.frame(predict(model_2, trainingset))
observed<-data.frame(trainingset$Yes)

#print(paste("Test"))
#print(trainingset)

#observed_test<-data.frame(testset$Yes)
#Id_test<-data.frame(mydatanew$Id)
print(cat(",Id,Observed,Predicted,MW","\n",file=output_filename))
for (index in 1:length(predicted[,1])){print(cat(index,Id_train[index,1],observed[index,1],predicted[index,1],trainingset[index,5],trainingset[index,9],trainingset[index,8],sep=" , ","\n",file=output_filename,append=TRUE))}

output_filename=paste(dir,impDescr,ext,sep="")
print("Entering function New_dataset")
New_dataset(model_2, output_filename, imp_val, fst_column,endpoint,tree,dir,dir2,Filename_2)
Predict_new(model_2, output_filename, imp_val, fst_column, endpoint, tree, dir, dir2, Filename_3)
print("done")

}	

rmse <- function(observed, predicted) {
sqrt( mean( (observed - predicted)^2 ))
}

# nearZeroVar <- function(dat) {
    # out <- lapply(dat, function(x) length(unique(x)))
    # want <- which(!out > 1)
    # unlist(want)
# }

 mae <- function(observed, predicted) {
 mean( abs(observed - predicted) )
 }


Predict_new <- function(model_2, output_filename, imp_val, fst_column, endpoint, tree, dir, dir2, Filename_3){
	ext=".csv"
	input=paste(dir,Filename_3,ext,sep="") 
    mydata <- data.frame(read.csv(file=input))
    mydataset <- mydata[ ,!(colnames(mydata) %in% c(fst_column))]
    mydataDescr <- data.frame(mydataset[ ,!(colnames(mydataset) %in% c(endpoint))])
    
    ImpVar <- data.frame(read.csv(file=output_filename))
	names(ImpVar) <- c("Descr","imp1","imp2")
	ImpVar <- ImpVar[order(ImpVar$imp2),]
	ts= paste(ImpVar[1:imp_val,1])
	data1 <- data.frame(mydata[,c(fst_column,"Chirality","CLOGP","NOCNT","HBD","MW","PSA","cPFLOGD","HBA","ROTB","Accptdonr","Boiling","Melting","Polarizability","Dielectric","Viscosity","DipoleM","Density","Surface","solvCLOGP","solvNOCNT","solvHBD","solvMW","solvPSA","solvHBA","solvROTB")])
	print(paste("impval = ", imp_val))
	
	
	mydatanew <- data.frame(mydataDescr)
    mydatanew <- cbind(fst_column = 0, mydatanew)
    mydatanew$Id<- mydata[,c(fst_column)]
    #mydatanew[endpoint] <- NA
    #mydatanew$Yes <- mydata[,c(endpoint)] 
    mydatanew <- data.frame(mydatanew)

	data2 <- data.frame(mydatanew[,c(ts)])
	#data2[endpoint] <- NA
	#data2$Yes <- mydata[,c(endpoint)]
	validationset <- data.frame(cbind(data1,data2))
	validationset <- data.frame(cbind(data1,data2))
	print (validationset)
    validationset <- validationset[ ,!(colnames(validationset) %in% c(fst_column))]
	predicted <- data.frame(predict(model_2, validationset))
	
	out2=paste("IndependentPredictionSet_",tree,"_",nrow(validationset),"_",filenum)
    output_filename=paste(dir2,out2,ext,sep="") 
    print (output_filename)
    
    print(cat(",Id,MW,Predicted","\n",file=output_filename))
    Id<-data.frame(mydata$Id)
    observed<-data.frame(mydata$Yes)
    for (index in 1:length(predicted[,1])){print(cat(index,Id[index,1],validationset[index,5],predicted[index,1],sep=" , ","\n",file=output_filename,append=TRUE))}
}


New_dataset <- function(model_2, output_filename, imp_val, fst_column,endpoint,tree, dir, dir2, Filename_2){
	ext=".csv"
	input=paste(dir,Filename_2,ext,sep="") 
    mydata <- data.frame(read.csv(file=input))
    mydataset <- mydata[ ,!(colnames(mydata) %in% c(fst_column))]
    mydataDescr <- data.frame(mydataset[ ,!(colnames(mydataset) %in% c(endpoint))])
    
    ImpVar <- data.frame(read.csv(file=output_filename))
	names(ImpVar) <- c("Descr","imp1","imp2")
	ImpVar <- ImpVar[order(ImpVar$imp2),]
	ts= paste(ImpVar[1:imp_val,1])
	data1 <- data.frame(mydata[,c(fst_column,"Chirality","CLOGP","NOCNT","HBD","MW","PSA","cPFLOGD","HBA","ROTB","Accptdonr","Boiling","Melting","Polarizability","Dielectric","Viscosity","DipoleM","Density","Surface","solvCLOGP","solvNOCNT","solvHBD","solvMW","solvPSA","solvHBA","solvROTB")])
	print(paste("impval = ", imp_val))
	
	
	mydatanew <- data.frame(mydataDescr)
    mydatanew <- cbind(fst_column = 0, mydatanew)
    mydatanew$Id<- mydata[,c(fst_column)]
    mydatanew[endpoint] <- NA
    mydatanew$Yes <- mydata[,c(endpoint)] 
    mydatanew <- data.frame(mydatanew)

	data2 <- data.frame(mydatanew[,c(ts)])
	data2[endpoint] <- NA
	data2$Yes <- mydata[,c(endpoint)]
	validationset <- data.frame(cbind(data1,data2))
	print (validationset)
    validationset <- validationset[ ,!(colnames(validationset) %in% c(fst_column))]
	predicted <- data.frame(predict(model_2, validationset))
	
	out2=paste("IndependentValidationSet_",tree,"_",nrow(validationset),"_",filenum)
    output_filename=paste(dir2,out2,ext,sep="") 
    print (output_filename)
    
    print(cat(",Id,Observed,Predicted","\n",file=output_filename))
    Id<-data.frame(mydata$Id)
    observed<-data.frame(mydata$Yes)
    for (index in 1:length(predicted[,1])){print(cat(index,Id[index,1],observed[index,1],predicted[index,1],sep=" , ","\n",file=output_filename,append=TRUE))}
}

for(filenum in 1:1000)
	RF_Modeling_test(filenum)
