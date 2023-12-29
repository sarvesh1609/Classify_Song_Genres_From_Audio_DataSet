# 1. First Stage :- All the necessary pacakages installations

install.packages("jsonlite")
install.packages("corrplot")
install.packages("caret")
install.packages("e1071")
install.packages("rpart")
install.packages("nnet")
install.packages( "yardstick")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("caTools")

############################## First Stage End ##################################################

#################################################################################################

# 2. Second stage :- loading the Libraries 

library(jsonlite)
library(corrplot)
library(caret)
library(e1071)
library(rpart)
library(nnet)
library(yardstick)
library(dplyr)
library(ggplot2)
library(caTools)

############################## Secound Stage End ##################################################

#################################################################################################

# 3. Third stage :- Load necessary data sets 

#we have two dataset. 
#1. CSV File :- basic information about the songs along with the genre (Total 16460 different songs)
#2. CSV File :- containing muscial features like danceability , acousticness , valence etc

# Song data - First file about the track along with the genre
Song_data <- read.csv("/Users/fma-rock-vs-hiphop.csv")

# Features data - features like danceability , acousticness , valence etc

Feature_data <- read.csv("/Users/echonest-metrics.csv")

############################## Third Stage End ##################################################

#################################################################################################

# 4. Fourth stage :- Data Exploration

# Lets Explore Song_data

print(head(song_data, 10))

# we have total 17734 rows 
total_row_count <- nrow(song_data)
print(total_row_count)

# we have total 17734 distinct rows 
distinct_rows <- unique(song_data)
count_distinct_rows <- nrow(distinct_rows)
print(count_distinct_rows)

# we have total 16460 distinct songs 
distinct_values <- unique(song_data$title)
total_distinct_values <- length(distinct_values)
print(total_distinct_values)

# Lets Explore Feature_data

# We have 9 varaibles, data type for track_id is int rest everything we have as number
str(Feature_data)

summary(Feature_data)
print(head(Feature_data, 10))

############################## Fourth Stage End ##################################################

#################################################################################################

# 5. Fifth stage :- Merge two data sets

# Merge both the dataset on common column track_id

Merge_Data <- merge(Feature_data, song_data[, c('track_id', 'genre_top')], by = 'track_id')

# we will be using this merge data set through our analaysis. 


############################## Fifth Stage End ##################################################

#################################################################################################

# 6. Sixth stage :- Explore merge data set.

# 1. Find the correlation amoung the varaibles. The highely correlated variables should 
#    be removed and reduce the varaibles if possible. 


# Calculate the correlation matrix
numeric_columns <- sapply(Merge_Data, is.numeric)
numeric_data <- Merge_Data[, numeric_columns]

# Calculate the correlation matrix
correlation_matrix <- cor(numeric_data)

# Print the correlation matrix
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
corrplot(
  correlation_matrix,
  method = "color",        # Use color to represent correlation
  #type = "upper",          # Show upper triangle of the matrix
  order = "hclust",        # Use hierarchical clustering to reorder variables
  tl.col = "black",        # Set text label color to black
  tl.srt = 45,             # Rotate text labels for better visibility
  addCoef.col = "black"     # Add numeric values to the plot with black color
)

# As no features exhibit a correlation of 0.8 or higher,
# I have determined that there are no highly correlated features. 
# Consequently, all variables or features will be considered for further analysis.

############################## Sixth Stage End ##################################################

#################################################################################################

# 7. Seventh stage :- Normalise merge data set.

# The next step is to normalize the data. our data can have various scale values present, 
# normalising the scale values helps preventing certain features from dominating others. 
# We will be using r scale function which uses Standardize features to normalise data. 

# First step would be creating two dataframes. 
# 1. With only numeric varaibles (8 varaibles) called as 'Predictors'.
# 2. With geners values (column name genre_top) called as 'Class'.

# Drop 'genre_top' and 'track_id' columns to create Predictors
Predictors <- Merge_Data[, !(names(Merge_Data) %in% c('genre_top', 'track_id'))]

# Create Class using the 'genre_top' column
Class <- Merge_Data$genre_top



# Secound step would be Standardize the features using the scale function.

# Standardize the features using the scale function
Scaled_train_Predictors <- scale(Predictors)

# Display the first few rows of the scaled features

head(Scaled_train_Predictors)

# Convert the scaled matrix to a data frame
Scaled_train_Predictors <- as.data.frame(Scaled_train_Predictors)


# We now have normalised values, but we still have one more issue that needs to be solved. 
# The issue is , we have too many features or varaibles which may lead to overfitting of model. 
# To reduce the variables or components we need to perform PCA (Principal component analysis)


############################## Seventh Stage End ##################################################

#################################################################################################

# 8. Eight stage :- Perform PCA (Principal component analysis) on Scaled_train_Predictors data.

# We have two steps in this process. 
# First step, we will apply PCA to our Scaled_train_Predictors data and check if we are able to drop 
# any varaible. 

# Second step, if first steps fails then we will apply PCA on cummulative data and check if any varaible
# dropping is observed or not. 


# First step

# Apply PCA
Pca_Outcome <- prcomp(Scaled_train_Predictors)

# Extract the explained variance ratios
Explained_variance <- Pca_Outcome$sdev^2 / sum(Pca_Outcome$sdev^2)

# Print the explained variance ratios

cat("Explained variance ratio : ", Explained_variance, "\n")

cat("Number of components =", length(Explained_variance))

# Plot the explained variance using a Scree Plot

plot(Explained_variance, type = "b", pch = 19, xlab = "Principal Component #", ylab = "Proportion of Variance Explained", main = "Scree Plot")

# Plot the explained variance using a barplot
barplot(Explained_variance, xlab = 'Principal Component #', ylab = 'Explained Variance', col = 'steelblue')


# Regrettably, the scree plot lacks a distinct elbow,
# making it challenging to determine the number of Fundamental dimensions using this method


# Second Step 

# plotting cumulative variance plot to reduce the components 

# Calculate the cumulative explained variance
cum_exp_variance <- cumsum(Explained_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90
plot(cum_exp_variance, type = "b", pch = 19, xlab = "Principal Component #", ylab = "Cumulative Explained Variance", main = "Cumulative Explained Variance Plot")
abline(h = 0.9, lty = 2, col = "red")

# from above graph we can say that only 6 componet or features are enough for modelling. 


# Therefore we use only 6 components. 

n_components <- 6
pca_result <- prcomp(Scaled_train_Predictors, center = TRUE, scale. = TRUE, rank. = n_components)

# Extract the principal component scores (projection)
pca_projection <- pca_result$x

# Get the dimensions (shape) of pca_projection
shape <- dim(pca_projection)

# Print the shape
cat("Shape of pca_projection:", shape, "\n")


# Now that we have reduced the features to six, 
# we proceed to build models using two algorithms: decision tree and logistic regression.
# To initiate this process, the initial step involves splitting our data
# into training and testing sets, which will be represented by variables
# named Train_Predictors, Test_Predictors, Train_Class, and Test_Class.

############################## Eight Stage End ##################################################

#################################################################################################

# 9. Nineth stage :- Split the data named Train_Predictors, Test_Predictors, Train_Class, and Test_Class.

# Set the seed for reproducibility
set.seed(10)

# Split the data
split <- sample.split(Class, SplitRatio = 0.7)
Train_Predictors <- pca_projection[split, ]
Test_Predictors <- pca_projection[!split, ]
Train_Class <- Class[split]
Test_Class <- Class[!split]


############################## Nineth Stage End ##################################################

#################################################################################################

# 10. Tenth stage :- Modelling.

# Here we have two models.
# A. Decision Tree
# B. Logistic Regression

# A. Decision Tree

# Train the decision tree
Decision_Tree <- rpart(Train_Class ~ ., data = as.data.frame(Train_Predictors), method = "class", control = rpart.control(seed = 10))

# Predict the labels for the test data
Predict_labels_for_Decision_Tree <- predict(Decision_Tree, as.data.frame(Test_Predictors), type = "class")


# Create Confusion Matrix and Statistics for desicion tree model


# Manually set the levels based on expected classes
expected_levels <- c("Hip-Hop", "Rock")  

# Set levels for both vectors
Predict_labels_for_Decision_Tree <- factor(Predict_labels_for_Decision_Tree, levels = expected_levels)
Test_Class <- factor(Test_Class, levels = expected_levels)

# Now create the confusion matrix
Confusion_Matrix_tree <- confusionMatrix(Test_Class, Predict_labels_for_Decision_Tree)

# Print the results

print("Decision Tree: \n")
print(Confusion_Matrix_tree)




# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
HipHop_Pridiction_Accuracy  <- Confusion_Matrix_tree$byClass['Pos Pred Value']
Rock_Pridiction_Accuracy    <- Confusion_Matrix_tree$byClass['Neg Pred Value']
Avg_Accuracy                <- Confusion_Matrix_tree$byClass['Balanced Accuracy']
precision                   <- Confusion_Matrix_tree$byClass['Precision']
recall                      <- Confusion_Matrix_tree$byClass['Recall']
f1_score                    <- Confusion_Matrix_tree$byClass['F1']
support                     <- Confusion_Matrix_tree$byClass['Support']
Sensitivity                 <- Confusion_Matrix_tree$byClass['Sensitivity']
Specificity                 <- Confusion_Matrix_tree$byClass['Specificity']

# Print the results

cat(" Decision-Tree HipHop Accuracy:", HipHop_Pridiction_Accuracy, "\n",
    "Decision-Tree Rock Accuracy:", Rock_Pridiction_Accuracy, "\n",
    "Decision-Tree Avg_Accuracy:", Avg_Accuracy, "\n",
    "Decision-Tree Precision:", precision, "\n",
    "Decision-Tree Recall:", recall, "\n",
    "Decision-Tree F1 Score:", f1_score, "\n",
    "Decision-Tree Sensitivity:", Sensitivity, "\n",
    "Decision-Tree Specificity:", Specificity, "\n")



# A. Logistic Regression


# Convert class labels to factors
Train_Class <- as.factor(Train_Class)
Test_Class <- as.factor(Test_Class)

# Train Multinomial Logistic Regression
multinom_model <- multinom(Train_Class ~ ., data = as.data.frame(Train_Predictors))

# Predict with Multinomial Logistic Regression
Predict_labels_for_Logistic_Regression <- predict(multinom_model, newdata = as.data.frame(Test_Predictors), type = "class")


# Create confusion matrix

Confusion_Matrix_Logistic_Regression <- confusionMatrix(Test_Class,Predict_labels_for_Logistic_Regression)



# Print the results

print("Logistic Regression: \n")
print(Confusion_Matrix_Logistic_Regression)



# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
HipHop_Pridiction_Accuracy  <- Confusion_Matrix_Logistic_Regression$byClass['Pos Pred Value']
Rock_Pridiction_Accuracy    <- Confusion_Matrix_Logistic_Regression$byClass['Neg Pred Value']
Avg_Accuracy                <- Confusion_Matrix_Logistic_Regression$byClass['Balanced Accuracy']
precision                   <- Confusion_Matrix_Logistic_Regression$byClass['Precision']
recall                      <- Confusion_Matrix_Logistic_Regression$byClass['Recall']
f1_score                    <- Confusion_Matrix_Logistic_Regression$byClass['F1']
support                     <- Confusion_Matrix_Logistic_Regression$byClass['Support']
Sensitivity                 <- Confusion_Matrix_Logistic_Regression$byClass['Sensitivity']
Specificity                 <- Confusion_Matrix_Logistic_Regression$byClass['Specificity']

# Print the results
cat(" Logistic_Regression HipHop Accuracy:", HipHop_Pridiction_Accuracy, "\n",
    "Logistic_Regression Rock Accuracy:", Rock_Pridiction_Accuracy, "\n",
    "Logistic_Regression Avg_Accuracy:", Avg_Accuracy, "\n",
    "Logistic_Regression Precision:", precision, "\n",
    "Logistic_Regression Recall:", recall, "\n",
    "Logistic_Regression F1 Score:", f1_score, "\n",
    "Logistic_Regression Support:", support, "\n",
    "Logistic_Regression F1 Score:", Sensitivity, "\n",
    "Logistic_Regression Support:", Specificity, "\n")

# By looking at both the model output it seems that the accuracy is almost same. 



############################## Tenth Stage End ##################################################

#################################################################################################

# 11. Eleventh stage :- Checking and fixing balance dataset.

# First step , check if data set is balanced or not.
# Second step, if not then fix it by sampling the data set.  

# Viewing the data- there is a class imbalance data. More Rock songs present in our
#dataset than Hip-Hop

# First step
# Create a count plot --Merge_Data
ggplot(Merge_Data, aes(x = genre_top)) +
  geom_bar() +
  labs(title = "Count Plot of genre_top", x = "genre_top", y = "Count")


# Subset only the hip-hop tracks
hip_hop <- Merge_Data[Merge_Data$genre_top == "Hip-Hop", ]

# Subset only the rock tracks
rock <- Merge_Data[Merge_Data$genre_top == "Rock", ]

# Get the dimensions of the hip-hop and rock data frame
dim(hip_hop)
dim(rock)

#3892 rock songs compared to only 910 hip hop songs. 
#Remove this imbalance by sampling the rock songs to have the same number as hip hop


# Second Step
# Assuming hip_hop and rock are data frames

# Set seed for reproducibility
set.seed(10)

# Sample rows from rock to match the number of rows in hip_hop
rock <- rock[sample(nrow(rock), nrow(hip_hop)), ]

# Assuming rock and hip_hop are data frames

# Concatenate to create the balanced dataset
balanced_data <- rbind(rock, hip_hop)

# Assuming balanced_data is a data frame
head(balanced_data)

# Now we have new dataset, perform all the steps from step 7 till step 10. 

############################## eleventh Stage End ##################################################

#################################################################################################


# 12. Twelfth stage :- perform all the steps from step 7 till step 10. 


# 7. Seventh stage :- Normalise merge data set.

# The next step is to normalize the data. our data can have various scale values present, 
# normalising the scale values helps preventing certain features from dominating others. 
# We will be using r scale function which uses Standardize features to normalise data. 

# First step would be creating two dataframes. 
# 1. With only numeric varaibles (8 varaibles) called as 'Predictors'.
# 2. With geners values (column name genre_top) called as 'Class'.

# Drop 'genre_top' and 'track_id' columns to create Predictors
Predictors <- balanced_data[, !(names(balanced_data) %in% c('genre_top', 'track_id'))]

# Create Class using the 'genre_top' column
Class <- balanced_data$genre_top



# Secound step would be Standardize the features using the scale function.

# Standardize the features using the scale function
Scaled_train_Predictors <- scale(Predictors)

# Display the first few rows of the scaled features

head(Scaled_train_Predictors)

# Convert the scaled matrix to a data frame
Scaled_train_Predictors <- as.data.frame(Scaled_train_Predictors)


# We now have normalised values, but we still have one more issue that needs to be solved. 
# The issue is , we have too many features or varaibles which may lead to overfitting of model. 
# To reduce the variables or components we need to perform PCA (Principal component analysis)


############################## Seventh Stage End ##################################################

#################################################################################################

# 8. Eight stage :- Perform PCA (Principal component analysis) on Scaled_train_Predictors data.

# We have two steps in this process. 
# First step, we will apply PCA to our Scaled_train_Predictors data and check if we are able to drop 
# any varaible. 

# Second step, if first steps fails then we will apply PCA on cummulative data and check if any varaible
# dropping is observed or not. 


# First step

# Apply PCA
Pca_Outcome <- prcomp(Scaled_train_Predictors)

# Extract the explained variance ratios
Explained_variance <- Pca_Outcome$sdev^2 / sum(Pca_Outcome$sdev^2)

# Print the explained variance ratios
print("Explained variance ratio : ")
print(Explained_variance)
cat("\n")
cat("Number of components =", length(Explained_variance))

# Plot the explained variance using a Scree Plot

plot(Explained_variance, type = "b", pch = 19, xlab = "Principal Component #", ylab = "Proportion of Variance Explained", main = "Scree Plot")

# Plot the explained variance using a barplot
barplot(Explained_variance, xlab = 'Principal Component #', ylab = 'Explained Variance', col = 'steelblue')


# Regrettably, the scree plot lacks a distinct elbow,
# making it challenging to determine the number of Fundamental dimensions using this method


# Second Step 

# plotting cumulative variance plot to reduce the components 

# Calculate the cumulative explained variance
cum_exp_variance <- cumsum(Explained_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90
plot(cum_exp_variance, type = "b", pch = 19, xlab = "Principal Component #", ylab = "Cumulative Explained Variance", main = "Cumulative Explained Variance Plot")
abline(h = 0.9, lty = 2, col = "red")

# from above graph we can say that only 6 componet or features are enough for modelling. 


# Therefore we use only 6 components. 

n_components <- 6
pca_result <- prcomp(Scaled_train_Predictors, center = TRUE, scale. = TRUE, rank. = n_components)

# Extract the principal component scores (projection)
pca_projection <- pca_result$x

# Get the dimensions (shape) of pca_projection
shape <- dim(pca_projection)

# Print the shape
cat("Shape of pca_projection:", shape, "\n")


# Now that we have reduced the features to six, 
# we proceed to build models using two algorithms: decision tree and logistic regression.
# To initiate this process, the initial step involves splitting our data
# into training and testing sets, which will be represented by variables
# named Train_Predictors, Test_Predictors, Train_Class, and Test_Class.

############################## Eight Stage End ##################################################

#################################################################################################

# 9. Nineth stage :- Split the data named Train_Predictors, Test_Predictors, Train_Class, and Test_Class.

# Set the seed for reproducibility
set.seed(10)

# Split the data
split <- sample.split(Class, SplitRatio = 0.7)
Train_Predictors <- pca_projection[split, ]
Test_Predictors  <- pca_projection[!split, ]
Train_Class      <- Class[split]
Test_Class       <- Class[!split]


############################## Nineth Stage End ##################################################

#################################################################################################

# 10. Tenth stage :- Modelling.

# Here we have two models.
# A. Decision Tree
# B. Logistic Regression

# A. Decision Tree

# Train the decision tree
Decision_Tree <- rpart(Train_Class ~ ., data = as.data.frame(Train_Predictors), method = "class", control = rpart.control(seed = 10))

# Predict the labels for the test data
Predict_labels_for_Decision_Tree <- predict(Decision_Tree, as.data.frame(Test_Predictors), type = "class")


# Create Confusion Matrix and Statistics for desicion tree model


# Manually set the levels based on expected classes
expected_levels <- c("Hip-Hop", "Rock")  

# Set levels for both vectors
Predict_labels_for_Decision_Tree <- factor(Predict_labels_for_Decision_Tree, levels = expected_levels)
Test_Class <- factor(Test_Class, levels = expected_levels)

# Now create the confusion matrix
Confusion_Matrix_tree <- confusionMatrix(Test_Class, Predict_labels_for_Decision_Tree)

# Print the results

print("Decision Tree: \n")
print(Confusion_Matrix_tree)




# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
HipHop_Pridiction_Accuracy  <- Confusion_Matrix_tree$byClass['Pos Pred Value']
Rock_Pridiction_Accuracy    <- Confusion_Matrix_tree$byClass['Neg Pred Value']
Avg_Accuracy                <- Confusion_Matrix_tree$byClass['Balanced Accuracy']
precision                   <- Confusion_Matrix_tree$byClass['Precision']
recall                      <- Confusion_Matrix_tree$byClass['Recall']
f1_score                    <- Confusion_Matrix_tree$byClass['F1']
support                     <- Confusion_Matrix_tree$byClass['Support']
Sensitivity                 <- Confusion_Matrix_tree$byClass['Sensitivity']
Specificity                 <- Confusion_Matrix_tree$byClass['Specificity']

# Print the results

cat(" Decision-Tree HipHop Accuracy:", HipHop_Pridiction_Accuracy, "\n",
    "Decision-Tree Rock Accuracy:", Rock_Pridiction_Accuracy, "\n",
    "Decision-Tree Avg_Accuracy:", Avg_Accuracy, "\n",
    "Decision-Tree Precision:", precision, "\n",
    "Decision-Tree Recall:", recall, "\n",
    "Decision-Tree F1 Score:", f1_score, "\n",
    "Decision-Tree Sensitivity:", Sensitivity, "\n",
    "Decision-Tree Specificity:", Specificity, "\n")



# A. Logistic Regression


# Convert class labels to factors
Train_Class <- as.factor(Train_Class)
Test_Class <- as.factor(Test_Class)

# Train Multinomial Logistic Regression
multinom_model <- multinom(Train_Class ~ ., data = as.data.frame(Train_Predictors))

# Predict with Multinomial Logistic Regression
Predict_labels_for_Logistic_Regression <- predict(multinom_model, newdata = as.data.frame(Test_Predictors), type = "class")


# Create confusion matrix

Confusion_Matrix_Logistic_Regression <- confusionMatrix(Test_Class,Predict_labels_for_Logistic_Regression)



# Print the results

print("Logistic Regression: \n")
print(Confusion_Matrix_Logistic_Regression)



# Extract Accuracy, precision, recall, F1 score,Sensitivity,Specificity and support
HipHop_Pridiction_Accuracy  <- Confusion_Matrix_Logistic_Regression$byClass['Pos Pred Value']
Rock_Pridiction_Accuracy    <- Confusion_Matrix_Logistic_Regression$byClass['Neg Pred Value']
Avg_Accuracy                <- Confusion_Matrix_Logistic_Regression$byClass['Balanced Accuracy']
precision                   <- Confusion_Matrix_Logistic_Regression$byClass['Precision']
recall                      <- Confusion_Matrix_Logistic_Regression$byClass['Recall']
f1_score                    <- Confusion_Matrix_Logistic_Regression$byClass['F1']
support                     <- Confusion_Matrix_Logistic_Regression$byClass['Support']
Sensitivity                 <- Confusion_Matrix_Logistic_Regression$byClass['Sensitivity']
Specificity                 <- Confusion_Matrix_Logistic_Regression$byClass['Specificity']

# Print the results
cat(" Logistic_Regression HipHop Accuracy:", HipHop_Pridiction_Accuracy, "\n",
    "Logistic_Regression Rock Accuracy:", Rock_Pridiction_Accuracy, "\n",
    "Logistic_Regression Avg_Accuracy:", Avg_Accuracy, "\n",
    "Logistic_Regression Precision:", precision, "\n",
    "Logistic_Regression Recall:", recall, "\n",
    "Logistic_Regression F1 Score:", f1_score, "\n",
    "Logistic_Regression Support:", support, "\n",
    "Logistic_Regression F1 Score:", Sensitivity, "\n",
    "Logistic_Regression Support:", Specificity, "\n")

# Now the prediction of hip-hop bad rock songs seems to be fine. 


############################## Twelfth Stage End ##################################################

#################################################################################################


# 13. Thirteen stage :- Model validation.

# Set seed for reproducibility
set.seed(1)

# Assuming labels is a vector representing the labels for your data
# Create 10-fold cross-validation folds
cv <- createFolds(Class, k = 10, list = TRUE, returnTrain = FALSE)

# Convert factor levels to valid R variable names
Test_Class <- as.factor(Test_Class)
levels(Test_Class) <- make.names(levels(Test_Class))

# Define the train control for cross-validation
train_control <- trainControl(method = "cv", number = 10, savePredictions = TRUE, classProbs = TRUE)

# Train the decision tree model with cross-validation
model_dt <- train(x = Test_Predictors, y = Test_Class, method = "rpart", trControl = train_control)

# Print the accuracy
cat("Decision Tree Classifier Accuracy:", mean(model_dt$resample$Accuracy))



#### logistic regression 

# Set seed for reproducibility
set.seed(1)

# Convert y_test to a factor if it's not already
y_test <- as.factor(Test_Class)

# Define the train control for cross-validation
train_control <- trainControl(method = "cv", number = 10, savePredictions = TRUE, classProbs = TRUE)

# Train the logistic regression model with cross-validation
model_lg <- train(x = Test_Predictors, y = Test_Class, method = "glm", trControl = train_control, family = "binomial")
# Print the accuracy
cat("Logistic Regression Accuracy:", mean(model_lg$resample$Accuracy))

# Hence Logistic Regression has better accuracy than decision tree. Therefore Logistic Regression model is used 
# to classify-song-genres-from-audio-data
