# https://spark.apache.org/docs/latest/mllib-decision-tree.html
# https://spark.apache.org/docs/latest/ml-classification-regression.html#logistic-regression

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier
import pandas as pd

spark = SparkSession.builder.appName('ml-bank').getOrCreate()

df = spark.read.csv('bank.csv', header=True, inferSchema=True)
print(pd.DataFrame(df.take(5), columns=df.columns).transpose())

# # vẽ biểu đồ tương quan
# sns.set(style="ticks", color_codes=True)
# sns.pairplot(df.toPandas(), hue="deposit")
# plt.show()


df = df.select('age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
               'loan', 'contact', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit')
cols = df.columns
categoricalColumns = ['job', 'marital', 'education',
                      'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(
        inputCol=categoricalCol, outputCol=categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[
                            categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol='deposit', outputCol='label')
stages += [label_stringIdx]

numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(df)
df = pipelineModel.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)

print(pd.DataFrame(df.take(5), columns=df.columns))

train, test = df.randomSplit([0.7, 0.3], seed=2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# In ra tập dữ liệu train 20 dòng đầu tiên
print("Train: ", train.show(20))

# LR
lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)
lrModel = lr.fit(train)

predictions = lrModel.transform(test)
print("Prediction LR: ", predictions.select('age', 'job', 'label',
      'rawPrediction', 'prediction', 'probability').show(10))

evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

# ID3
dt = DecisionTreeClassifier(featuresCol='features',
                            labelCol='label', maxDepth=3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
print("Prediction ID3: ", predictions.select('age', 'job', 'label',
      'rawPrediction', 'prediction', 'probability').show(10))

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions,
      {evaluator.metricName: "areaUnderROC"})))
