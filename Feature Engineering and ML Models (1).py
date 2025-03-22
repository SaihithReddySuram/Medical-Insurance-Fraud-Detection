# Databricks notebook source
from pyspark.sql.functions import col, explode, array, lit
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# COMMAND ----------

# Read the dataset into a DataFrame
df = spark.read.csv("dbfs:/FileStore/shared_uploads/ssuram@gmu.edu/medical_insurance_Dataset.csv", header=True, inferSchema=True)

# Display the DataFrame
display(df)

# COMMAND ----------

df = df.withColumn("Tot_Sum_Clms", df["Tot_Sum_Clms"].cast(IntegerType()))
df = df.withColumn("Tot_Avg_Clms", df["Tot_Avg_Clms"].cast(IntegerType()))
df = df.withColumn("Tot_Max_Clms", df["Tot_Max_Clms"].cast(IntegerType()))
df = df.withColumn("Tot_Sum_Day_Suply", df["Tot_Sum_Day_Suply"].cast(IntegerType()))
df = df.withColumn("Tot_Avg_Day_Suply", df["Tot_Avg_Day_Suply"].cast(IntegerType()))
df = df.withColumn("Tot_Max_Day_Suply", df["Tot_Max_Day_Suply"].cast(IntegerType()))
df = df.withColumn("Tot_Sum_Drug_Cst", df["Tot_Sum_Drug_Cst"].cast(IntegerType()))
df = df.withColumn("Tot_Avg_Drug_Cst", df["Tot_Avg_Drug_Cst"].cast(IntegerType()))
df = df.withColumn("Tot_Max_Drug_Cst", df["Tot_Max_Drug_Cst"].cast(IntegerType()))
df = df.withColumn("Tot_Sum_30day_Fills", df["Tot_Sum_30day_Fills"].cast(IntegerType()))
df = df.withColumn("Tot_Avg_30day_Fills", df["Tot_Avg_30day_Fills"].cast(IntegerType()))
df = df.withColumn("Tot_Max_30day_Fills", df["Tot_Max_30day_Fills"].cast(IntegerType()))
df = df.withColumn("Tot_Sum_Benes", df["Tot_Sum_Benes"].cast(IntegerType()))
df = df.withColumn("Tot_Avg_Benes", df["Tot_Avg_Benes"].cast(IntegerType()))
df = df.withColumn("Tot_Max_Benes", df["Tot_Max_Benes"].cast(IntegerType()))
df = df.withColumn("Total_Amount_of_Payment_USDollars", df["Total_Amount_of_Payment_USDollars"].cast(IntegerType()))

# COMMAND ----------

df = df.filter(((df.Total_Amount_of_Payment_USDollars > 0) & (df.Fraud == 0)) | (df.Fraud == 1))

# COMMAND ----------

df

# COMMAND ----------

value_counts = df.groupBy('Fraud').count()

# Display the value counts
value_counts.show()

# COMMAND ----------

major_df = df.filter(col("Fraud") == 0)
minor_df = df.filter(col("Fraud") == 1)
ratio = int(major_df.count()/minor_df.count())
print("ratio: {}".format(ratio))

# COMMAND ----------

a = range(ratio)
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows 
oversampled_combined_df = major_df.unionAll(oversampled_df)
oversampled_combined_df.display()

# COMMAND ----------

oversampled_combined_df.count()

# COMMAND ----------

value_counts = oversampled_combined_df.groupBy('Fraud').count()

# Display the value counts
value_counts.show()

# COMMAND ----------

sampled_majority_df = major_df.sample(False, 1/ratio)
undersampled_combined_df_2 = sampled_majority_df.unionAll(minor_df)
undersampled_combined_df_2.display()

# COMMAND ----------

value_counts = undersampled_combined_df_2.groupBy('Fraud').count()

# Display the value counts
value_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC OverSampling Models

# COMMAND ----------

trainDF, testDF = oversampled_combined_df.randomSplit([0.8, 0.2], seed=42)
print(trainDF.cache().count()) # Cache because accessing training data multiple times
print(testDF.count())

# COMMAND ----------

value_counts = testDF.groupBy('Fraud').count()

# Display the value counts
value_counts.show()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
 
categoricalCols = ["Prscrbr_State", "Prscrbr_Type", "Max_Tot_Clms_Brand", "Max_Tot_Day_Suply_Brand", "Max_Tot_Drug_Cst_Brand"]
 
# The following two lines are estimators. They return functions that we will later apply to transform the dataset.
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols], handleInvalid = "keep") 
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categoricalCols]) 
 

# COMMAND ----------

stringIndexerModel = stringIndexer.fit(trainDF)
display(stringIndexerModel.transform(trainDF))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
 
# This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
numericCols = ["Tot_Sum_Clms" ,"Tot_Avg_Clms" ,"Tot_Max_Clms" ,"Tot_Sum_Day_Suply" ,"Tot_Avg_Day_Suply" ,"Tot_Max_Day_Suply" ,"Tot_Sum_Drug_Cst" ,"Tot_Avg_Drug_Cst" ,"Tot_Max_Drug_Cst" ,"Tot_Sum_30day_Fills" ,"Tot_Avg_30day_Fills" ,"Tot_Max_30day_Fills" ,"Tot_Sum_Benes" ,"Tot_Avg_Benes" ,"Tot_Max_Benes" ,"Total_Amount_of_Payment_USDollars"]
assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC LogisticRegression

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
 
lr = LogisticRegression(featuresCol="features", labelCol="Fraud", regParam=1.0)

# COMMAND ----------

# Define the pipeline based on the stages created in previous steps.
pipeline = Pipeline(stages=[stringIndexer, encoder, vecAssembler, lr])
 
# Define the pipeline model.
pipelineModel = pipeline.fit(trainDF)
 
# Apply the pipeline model to the test dataset.
predDF = pipelineModel.transform(testDF)


# COMMAND ----------

display(predDF.select("features", "Fraud", "prediction", "probability"))

# COMMAND ----------

display(pipelineModel.stages[-1], predDF.drop("prediction", "rawPrediction", "probability"), "ROC")

# COMMAND ----------

bcEvaluator = BinaryClassificationEvaluator(labelCol="Fraud" ,metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predDF)}")
 
mcEvaluator = MulticlassClassificationEvaluator(labelCol="Fraud" ,metricName="accuracy")
print(f"Accuracy: {mcEvaluator.evaluate(predDF)}")

# COMMAND ----------

len(weights)

# COMMAND ----------

# MAGIC %md
# MAGIC DecisionTreeClassifier

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(labelCol="Fraud", featuresCol="features")

# COMMAND ----------

pipeline = Pipeline(stages=[stringIndexer, encoder, vecAssembler, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainDF)

# Make predictions.
predictions = model.transform(testDF)

# COMMAND ----------

display(model.stages[-1], predictions.drop("prediction", "rawPrediction", "probability"), "ROC")

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(
    labelCol="Fraud", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# COMMAND ----------


from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
bcEvaluator = BinaryClassificationEvaluator(labelCol="Fraud" ,metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predictions)}")

# COMMAND ----------

# MAGIC %md
# MAGIC Random Forest

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="Fraud", featuresCol="features")

pipeline = Pipeline(stages=[stringIndexer, encoder, vecAssembler, rf])

# COMMAND ----------

# Train model.  This also runs the indexers.
model = pipeline.fit(trainDF)

# Make predictions.
predictions = model.transform(testDF)

# COMMAND ----------

evaluator = MulticlassClassificationEvaluator(
    labelCol="Fraud", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")

# COMMAND ----------

bcEvaluator = BinaryClassificationEvaluator(labelCol="Fraud" ,metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predictions)}")

# COMMAND ----------

bestPipeline = model
bestModel = bestPipeline.stages[-1]
importances = bestModel.featureImportances

feature_list = [c + "OHE" for c in categoricalCols] + numericCols
#print(feature_list)
print("Feature Importances:")
for feature, importance in zip(feature_list, importances):
    print(f"{feature}: {importance:.4f}")

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(modelType="multinomial" ,labelCol="Fraud")

pipeline = Pipeline(stages=[stringIndexer, encoder, vecAssembler, nb])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainDF)

# Make predictions.
predictions = model.transform(testDF)

# COMMAND ----------

bcEvaluator = BinaryClassificationEvaluator(labelCol="Fraud" ,metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predictions)}")
 
mcEvaluator = MulticlassClassificationEvaluator(labelCol="Fraud" ,metricName="accuracy")
print(f"Accuracy: {mcEvaluator.evaluate(predictions)}")

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier

gb = GBTClassifier(labelCol = 'Fraud', featuresCol="features", maxIter = 3, maxDepth=2)

pipeline = Pipeline(stages=[stringIndexer, encoder, vecAssembler, gb])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainDF)

# Make predictions.
predictions = model.transform(testDF)

# COMMAND ----------

bcEvaluator = BinaryClassificationEvaluator(labelCol="Fraud" ,metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predictions)}")
 
mcEvaluator = MulticlassClassificationEvaluator(labelCol="Fraud" ,metricName="accuracy")
print(f"Accuracy: {mcEvaluator.evaluate(predictions)}")

# COMMAND ----------

# DBTITLE 0,UnderSampling
# MAGIC %md
# MAGIC UnderSamping Models

# COMMAND ----------

trainDF, testDF = undersampled_combined_df_2.randomSplit([0.8, 0.2], seed=42)
print(trainDF.cache().count()) # Cache because accessing training data multiple times
print(testDF.count())

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
 
categoricalCols = ["Prscrbr_State", "Prscrbr_Type", "Max_Tot_Clms_Brand", "Max_Tot_Day_Suply_Brand", "Max_Tot_Drug_Cst_Brand"]
 
# The following two lines are estimators. They return functions that we will later apply to transform the dataset.
stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=[x + "Index" for x in categoricalCols], handleInvalid = "keep") 
encoder = OneHotEncoder(inputCols=stringIndexer.getOutputCols(), outputCols=[x + "OHE" for x in categoricalCols]) 
 

# COMMAND ----------

stringIndexerModel = stringIndexer.fit(trainDF)
display(stringIndexerModel.transform(trainDF))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
 
# This includes both the numeric columns and the one-hot encoded binary vector columns in our dataset.
numericCols = ["Tot_Sum_Clms" ,"Tot_Avg_Clms" ,"Tot_Max_Clms" ,"Tot_Sum_Day_Suply" ,"Tot_Avg_Day_Suply" ,"Tot_Max_Day_Suply" ,"Tot_Sum_Drug_Cst" ,"Tot_Avg_Drug_Cst" ,"Tot_Max_Drug_Cst" ,"Tot_Sum_30day_Fills" ,"Tot_Avg_30day_Fills" ,"Tot_Max_30day_Fills" ,"Tot_Sum_Benes" ,"Tot_Avg_Benes" ,"Tot_Max_Benes" ,"Total_Amount_of_Payment_USDollars"]
assemblerInputs = [c + "OHE" for c in categoricalCols] + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

from pyspark.ml import Pipeline
 
# Define the pipeline based on the stages created in previous steps.
pipeline = Pipeline(stages=[stringIndexer, encoder, vecAssembler, lr])
 
# Define the pipeline model.
pipelineModel = pipeline.fit(trainDF)
 
# Apply the pipeline model to the test dataset.
predDF = pipelineModel.transform(testDF)


# COMMAND ----------

display(predDF.select("features", "Fraud", "prediction", "probability"))

# COMMAND ----------

display(pipelineModel.stages[-1], predDF.drop("prediction", "rawPrediction", "probability"), "ROC")

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
 
bcEvaluator = BinaryClassificationEvaluator(labelCol="Fraud" ,metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predDF)}")
 
mcEvaluator = MulticlassClassificationEvaluator(labelCol="Fraud" ,metricName="accuracy")
print(f"Accuracy: {mcEvaluator.evaluate(predDF)}")

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
 
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .build())

# Create a 3-fold CrossValidator
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=bcEvaluator, numFolds=3, parallelism = 4)


# COMMAND ----------

# Run cross validations. This step takes a few minutes and returns the best model found from the cross validation.
cvModel = cv.fit(trainDF)

# Use the model identified by the cross-validation to make predictions on the test dataset
cvPredDF = cvModel.transform(testDF)
 
# Evaluate the model's performance based on area under the ROC curve and accuracy 
print(f"Area under ROC curve: {bcEvaluator.evaluate(cvPredDF)}")
print(f"Accuracy: {mcEvaluator.evaluate(cvPredDF)}")

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(labelCol="Fraud", featuresCol="features")

pipeline = Pipeline(stages=[stringIndexer, encoder, vecAssembler, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainDF)

# Make predictions.
predictions = model.transform(testDF)

evaluator = MulticlassClassificationEvaluator(
    labelCol="Fraud", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
bcEvaluator = BinaryClassificationEvaluator(labelCol="Fraud" ,metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predictions)}")

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol="Fraud", featuresCol="features")

pipeline = Pipeline(stages=[stringIndexer, encoder, vecAssembler, rf])

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Define the hyperparameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# Create the cross-validator
cross_validator = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="Fraud", metricName="accuracy"),
                          numFolds=5, seed=42)

# Train the model with the best hyperparameters
cv_model = cross_validator.fit(trainDF)

predictions = cv_model.transform(testDF)

print(f"Area under ROC curve: {bcEvaluator.evaluate(predictions)}")
print(f"Accuracy: {mcEvaluator.evaluate(predictions)}")

# COMMAND ----------

from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(modelType="multinomial" ,labelCol="Fraud")

pipeline = Pipeline(stages=[stringIndexer, encoder, vecAssembler, nb])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainDF)

# Make predictions.
predictions = model.transform(testDF)

# COMMAND ----------

bcEvaluator = BinaryClassificationEvaluator(labelCol="Fraud" ,metricName="areaUnderROC")
print(f"Area under ROC curve: {bcEvaluator.evaluate(predictions)}")
 
mcEvaluator = MulticlassClassificationEvaluator(labelCol="Fraud" ,metricName="accuracy")
print(f"Accuracy: {mcEvaluator.evaluate(predictions)}")

# COMMAND ----------


