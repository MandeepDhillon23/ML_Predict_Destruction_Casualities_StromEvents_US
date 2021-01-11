#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Start time noted for Cloud vs Stand-alone comparison
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)


# In[2]:


from pyspark.ml.regression import LinearRegression
import datetime
from datetime import datetime
from dateutil.parser import parse
import pyspark
import pandas as pd
from pyspark.sql.functions import to_timestamp
from pyspark.sql import functions as sf
from pyspark.sql.functions import unix_timestamp
from pyspark.sql.functions import from_unixtime
from pyspark.sql import functions as F
from pyspark.sql.functions import *
import numpy as np

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


# In[3]:


# Reads all .csv files that are StormEvent files into a dataframe
dfOriginal = spark.read.csv('/FileStore/tables/StormEvents*', header="true", inferSchema="true")

# Creates a summary dataframe 'df' using predetermined features believed to be useful
df = dfOriginal.select('EVENT_ID', 'STATE_FIPS', 'STATE', 'EVENT_TYPE', 'BEGIN_DATE_TIME', 'END_DATE_TIME', 'INJURIES_DIRECT', 'INJURIES_INDIRECT', 'DEATHS_DIRECT', 'DEATHS_INDIRECT', 'DAMAGE_PROPERTY', 'DAMAGE_CROPS', 'MAGNITUDE', 'MAGNITUDE_TYPE', 'BEGIN_LAT', 'BEGIN_LON', 'END_LAT', 'END_LON', 'TOR_F_SCALE', 'TOR_LENGTH', 'TOR_WIDTH')

# Summing up Casualties in a new column. Filtering out "K" in costs and "M" is replace with "000" to indicate millions. Label column titled "CASUALTIES" is created.
df = df.withColumn('DAMAGE_CROPS', regexp_replace('DAMAGE_CROPS', 'K', ''))
df = df.withColumn('DAMAGE_CROPS', regexp_replace('DAMAGE_CROPS', 'M', '000'))
df = df.withColumn('DAMAGE_PROPERTY', regexp_replace('DAMAGE_PROPERTY', 'K', ''))
df = df.withColumn('DAMAGE_PROPERTY', regexp_replace('DAMAGE_PROPERTY', 'M', '000'))
df = df.withColumn('CASUALTIES', df.INJURIES_INDIRECT + df.INJURIES_DIRECT + df.DEATHS_DIRECT + df.DEATHS_INDIRECT)
df = df.withColumn('TOR_F_SCALE', regexp_replace('TOR_F_SCALE', 'EF', ''))

# Damage columns are converted to integers and summed into a label column titled "COST".
df = df.withColumn('TOR_F_SCALE', df.TOR_F_SCALE.cast('int'))
df = df.withColumn('DAMAGE_CROPS', df.DAMAGE_CROPS.cast('int'))
df = df.withColumn('DAMAGE_PROPERTY', df.DAMAGE_CROPS.cast('int'))
df = df.withColumn('STATE_FIPS', df.STATE_FIPS.cast('int'))
df = df.withColumn('COST', df.DAMAGE_PROPERTY + df.DAMAGE_CROPS)

# Creating time feature columns of "EVENT_LENGTH_HOURS" and "YEAR"
timeFmt = "dd-MMM-yy HH:mm:ss"
year = (F.unix_timestamp('BEGIN_DATE_TIME', format=timeFmt)) / 60 / 60 / 24 / 365 + 1970
timeDiff = (F.unix_timestamp('END_DATE_TIME', format=timeFmt) - F.unix_timestamp('BEGIN_DATE_TIME', format=timeFmt))/3600
df = df.withColumn("EVENT_LENGTH_HOURS", timeDiff)
df = df.withColumn("YEAR", year)

# Distance storm event traveled
df = df.withColumn("A", ( sin((df.END_LAT-df.BEGIN_LAT)*(3.1415/180)/2)**2 + cos(df.BEGIN_LAT * (3.1415/180)) * cos(df.END_LAT * (3.1415/180)) * (sin((df.END_LON - df.BEGIN_LON) * (3.1415/180))**2)))
df = df.withColumn("C", 2 * atan2((df.A)**(.5), (1-df.A)**(.5)))
df = df.withColumn("DISTANCE_TRAVELED", df.C * 6373)

#df.select('EVENT_TYPE').distinct().show(100)


# In[4]:


#<<---TORNADO DATA--->>
# Creating dataframe for tornado data with features specific to tornado storm events (TOR_F_SCALE, TOR_LENGTH, TOR_WIDTH)
dfTorn = df.select('EVENT_ID', 'STATE_FIPS', 'STATE', 'EVENT_TYPE','YEAR', 'TOR_F_SCALE', 'TOR_LENGTH', 'TOR_WIDTH', 'CASUALTIES','COST','EVENT_LENGTH_HOURS','DISTANCE_TRAVELED')
dfTorn = dfTorn.where(dfTorn.EVENT_TYPE == 'Tornado')
dfTorn = dfTorn.select('STATE_FIPS', 'YEAR', 'TOR_F_SCALE', 'TOR_LENGTH', 'TOR_WIDTH', 'EVENT_LENGTH_HOURS','DISTANCE_TRAVELED','CASUALTIES','COST')
dfTorn = dfTorn.na.drop()
#dfTorn.show()

# Creating Scatter Matrix plot to visualize correlation
numeric_features = [t[0] for t in dfTorn.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = dfTorn.select(numeric_features).sample(False, 0.8).toPandas()
axs = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
display()


# In[5]:


#<<---TORNADO CORRELATION DATA--->>
#Calculate correlation coefficients for data.
import six
for i in dfTorn.columns:
    if not( isinstance(dfTorn.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to COST for ", i, dfTorn.stat.corr('COST',i))

for i in dfTorn.columns:
  if not( isinstance(dfTorn.select(i).take(1)[0][0], six.string_types)):
    print( "Correlation to CASUALTIES for ", i, dfTorn.stat.corr('CASUALTIES',i))


# In[6]:


#<<---TORNADO COST REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS','YEAR', 'TOR_F_SCALE', 'TOR_LENGTH', 'TOR_WIDTH'], outputCol = 'features')
vdfTorn = vectorAssembler.transform(dfTorn)
vdfTorn = vdfTorn.select(['features', 'COST'])
#vdfTorn.show(3)

splits = vdfTorn.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='COST', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","COST","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="COST",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","COST","features").show()


# In[7]:


#<<---TORNADO CASUALTIES REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS','YEAR', 'TOR_F_SCALE', 'TOR_LENGTH', 'TOR_WIDTH'], outputCol = 'features')
vdfTorn = vectorAssembler.transform(dfTorn)
vdfTorn = vdfTorn.select(['features', 'CASUALTIES'])
#vdfTorn.show(3)

splits = vdfTorn.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='CASUALTIES', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","CASUALTIES","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="CASUALTIES",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","CASUALTIES","features").show()


# In[8]:


#<<---COLD/WIND CHILL DATA--->>
dfWntStrm = df.select('EVENT_ID', 'STATE_FIPS', 'STATE', 'EVENT_TYPE','YEAR', 'CASUALTIES','COST','EVENT_LENGTH_HOURS')
dfWntStrm = dfWntStrm.where(dfWntStrm.EVENT_TYPE == 'Cold/Wind Chill')

dfWntStrm = dfWntStrm.na.drop()
dfWntStrm.show(5)

numeric_features = [t[0] for t in dfWntStrm.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = dfWntStrm.select(numeric_features).sample(False, 0.8).toPandas()
axs = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
display()


# In[9]:


#<<---COLD/WIND CHILL CORRELATION DATA--->>
#Calculate correlation coefficients for data.
import six
for i in dfWntStrm.columns:
    if not( isinstance(dfWntStrm.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to COST for ", i, dfWntStrm.stat.corr('COST',i))
        
for i in dfWntStrm.columns:
  if not( isinstance(dfWntStrm.select(i).take(1)[0][0], six.string_types)):
    print( "Correlation to CASUALTIES for ", i, dfWntStrm.stat.corr('CASUALTIES',i))


# In[10]:


#<<---COLD/WIND CHILL COST REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR','EVENT_LENGTH_HOURS'], outputCol = 'features')
vdfWntStrm = vectorAssembler.transform(dfWntStrm)
vdfWntStrm = vdfWntStrm.select(['features', 'COST'])
#vdfTorn.show(3)

splits = vdfWntStrm.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='COST', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","COST","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="COST",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","COST","features").show()


# In[11]:


#<<---COLD/WIND CHILL CASUALTIES REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR','EVENT_LENGTH_HOURS'], outputCol = 'features')
vdfWntStrm = vectorAssembler.transform(dfWntStrm)
vdfWntStrm = vdfWntStrm.select(['features', 'CASUALTIES'])
#vdfTorn.show(3)

splits = vdfWntStrm.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='CASUALTIES', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","CASUALTIES","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="CASUALTIES",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","CASUALTIES","features").show()


# In[12]:


#<<---BLIZZARD DATA--->>
dfBliz = df.select('EVENT_ID', 'STATE_FIPS', 'STATE', 'EVENT_TYPE', 'YEAR', 'COST', 'CASUALTIES', 'EVENT_LENGTH_HOURS')
dfBliz = dfBliz.where(dfBliz.EVENT_TYPE == 'Blizzard')
dfBliz = dfBliz.na.drop()
dfBliz.show(5)

numeric_features = [t[0] for t in dfBliz.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = dfBliz.select(numeric_features).sample(False, 0.8).toPandas()
axs = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
display()


# In[13]:


#<<---BLIZZARD CORRELATION DATA--->>
#Calculate correlation coefficients for data.
import six
for i in dfBliz.columns:
    if not( isinstance(dfBliz.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to COST for ", i, dfBliz.stat.corr('COST',i))
        
for i in dfBliz.columns:
  if not( isinstance(dfBliz.select(i).take(1)[0][0], six.string_types)):
    print( "Correlation to CASUALTIES for ", i, dfBliz.stat.corr('CASUALTIES',i))


# In[14]:


#<<---BLIZZARD COST REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR', 'EVENT_LENGTH_HOURS'], outputCol = 'features')
vdfBliz = vectorAssembler.transform(dfBliz)
vdfBliz = vdfBliz.select(['features', 'COST'])
#vdfBliz.show(3)

splits = vdfBliz.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='COST', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","COST","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="COST",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","COST","features").show()


# In[15]:


#<<---BLIZZARD CASUALTIES REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR', 'EVENT_LENGTH_HOURS'], outputCol = 'features')
vdfBliz = vectorAssembler.transform(dfBliz)
vdfBliz = vdfBliz.select(['features', 'CASUALTIES'])
#vdfBliz.show(3)

splits = vdfBliz.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='CASUALTIES', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","CASUALTIES","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="CASUALTIES",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","CASUALTIES","features").show()


# In[16]:


#<<---HURRICANE DATA--->>
dfHrcn = df.select('EVENT_ID', 'STATE_FIPS', 'STATE', 'EVENT_TYPE', 'YEAR', 'COST', 'CASUALTIES', 'EVENT_LENGTH_HOURS')
dfHrcn = dfHrcn.where(dfHrcn.EVENT_TYPE == 'Hurricane')
dfHrcn = dfHrcn.na.drop()

dfHrcn.show(5)

numeric_features = [t[0] for t in dfHrcn.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = dfHrcn.select(numeric_features).sample(False, 0.8).toPandas()
axs = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
display()


# In[17]:


#<<---HURRICANE CORRELATION DATA--->>
#Calculate correlation coefficients for data.
import six
for i in dfHrcn.columns:
    if not( isinstance(dfHrcn.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to COST for ", i, dfHrcn.stat.corr('COST',i))
        
for i in dfHrcn.columns:
  if not( isinstance(dfHrcn.select(i).take(1)[0][0], six.string_types)):
    print( "Correlation to CASUALTIES for ", i, dfHrcn.stat.corr('CASUALTIES',i))


# In[18]:


#<<---HURRICANE COST REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR', 'EVENT_LENGTH_HOURS'], outputCol = 'features')
vdfHrcn = vectorAssembler.transform(dfHrcn)
vdfHrcn = vdfHrcn.select(['features', 'COST'])
#vdfTorn.show(3)

splits = vdfHrcn.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='COST', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","COST","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="COST",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","COST","features").show()


# In[19]:


#<<---HURRICANE CASUALTIES REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR', 'EVENT_LENGTH_HOURS'], outputCol = 'features')
vdfHrcn = vectorAssembler.transform(dfHrcn)
vdfHrcn = vdfHrcn.select(['features', 'CASUALTIES'])
#vdfTorn.show(3)

splits = vdfHrcn.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='CASUALTIES', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","CASUALTIES","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="CASUALTIES",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","CASUALTIES","features").show()


# In[20]:


#<<---FLOOD DATA--->>
dfFld = df.select('EVENT_ID', 'STATE_FIPS', 'STATE', 'EVENT_TYPE', 'YEAR', 'EVENT_LENGTH_HOURS', 'COST','CASUALTIES','DISTANCE_TRAVELED')
dfFld = dfFld.where(dfFld.EVENT_TYPE == 'Flood')
dfFld = dfFld.na.drop()

dfFld.show(5)

numeric_features = [t[0] for t in dfFld.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = dfFld.select(numeric_features).sample(False, 0.8).toPandas()
axs = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
display()


# In[21]:


#<<---FLOOD CORRELATION DATA--->>
#Calculate correlation coefficients for data.
import six
for i in dfFld.columns:
    if not( isinstance(dfFld.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to COST for ", i, dfFld.stat.corr('COST',i))
        
for i in dfFld.columns:
  if not( isinstance(dfFld.select(i).take(1)[0][0], six.string_types)):
    print( "Correlation to CASUALTIES for ", i, dfFld.stat.corr('CASUALTIES',i))


# In[22]:


#<<---FLOOD COST REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR', 'EVENT_LENGTH_HOURS','DISTANCE_TRAVELED'], outputCol = 'features')
vdfFld = vectorAssembler.transform(dfFld)
vdfFld = vdfFld.select(['features', 'COST'])
#vdfTorn.show(3)

splits = vdfFld.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='COST', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","COST","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="COST",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","COST","features").show()


# In[23]:


#<<---FLOOD CASUALTIES REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR', 'EVENT_LENGTH_HOURS','DISTANCE_TRAVELED'], outputCol = 'features')
vdfFld = vectorAssembler.transform(dfFld)
vdfFld = vdfFld.select(['features', 'CASUALTIES'])
#vdfTorn.show(3)

splits = vdfFld.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='CASUALTIES', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","CASUALTIES","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="CASUALTIES",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","CASUALTIES","features").show()


# In[24]:


#<<---FLASH FLOOD DATA--->>
dfFlshFld = df.select('EVENT_ID', 'STATE_FIPS', 'STATE', 'EVENT_TYPE', 'YEAR','EVENT_LENGTH_HOURS','COST','CASUALTIES','DISTANCE_TRAVELED')
dfFlshFld = dfFlshFld.where(dfFlshFld.EVENT_TYPE == 'Flash Flood')
dfFld = dfFld.na.drop()

dfFlshFld.show(5)

numeric_features = [t[0] for t in dfFlshFld.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = dfFlshFld.select(numeric_features).sample(False, 0.8).toPandas()
axs = pd.plotting.scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())
display()


# In[25]:


#<<---FLASH CORRELATION DATA--->>
#Calculate correlation coefficients for data.
import six
for i in dfFlshFld.columns:
    if not( isinstance(dfFlshFld.select(i).take(1)[0][0], six.string_types)):
        print( "Correlation to COST for ", i, dfFlshFld.stat.corr('COST',i))
        
for i in dfFlshFld.columns:
  if not( isinstance(dfFlshFld.select(i).take(1)[0][0], six.string_types)):
    print( "Correlation to CASUALTIES for ", i, dfFlshFld.stat.corr('CASUALTIES',i))


# In[26]:


#<<---FLASH FLOOD COST REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR','EVENT_LENGTH_HOURS', 'DISTANCE_TRAVELED'], outputCol = 'features')
vdfFlshFld = vectorAssembler.transform(dfFlshFld)
vdfFlshFld = vdfFlshFld.select(['features', 'COST'])
#vdfTorn.show(3)

splits = vdfFlshFld.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='COST', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","COST","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="COST",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","COST","features").show()


# In[27]:


#<<---FLASH FLOOD CASUALTIES REGRESSION MODEL--->>
from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols = ['STATE_FIPS', 'YEAR','EVENT_LENGTH_HOURS', 'DISTANCE_TRAVELED'], outputCol = 'features')
vdfFlshFld = vectorAssembler.transform(dfFlshFld)
vdfFlshFld = vdfFlshFld.select(['features', 'CASUALTIES'])
#vdfTorn.show(3)

splits = vdfFlshFld.randomSplit([0.75, 0.25])
train_df = splits[0]
test_df = splits[1]

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol = 'features', labelCol='CASUALTIES', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Linear Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

trainingSummary = lr_model.summary
print("Training Data RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("Training r2: %f" % trainingSummary.r2)

lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","CASUALTIES","features").show(5)
from pyspark.ml.evaluation import RegressionEvaluator
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="CASUALTIES",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

predictions = lr_model.transform(test_df)
predictions.select("prediction","CASUALTIES","features").show()


# In[28]:


#Start time noted for Cloud vs Stand-alone comparison
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)


# In[29]:




