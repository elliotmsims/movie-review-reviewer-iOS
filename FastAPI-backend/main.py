from typing import Optional
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizerModel
from pyspark.ml.classification import LogisticRegressionModel
from fastapi import FastAPI


# load models
ss = SparkSession.builder.appName("MovieReviewer").getOrCreate()
cv_model = CountVectorizerModel.load("../cloud-ML/trained-models/cv")
lr_model = LogisticRegressionModel.load("../cloud-ML/trained-models/lr")

# create FastAPI app
app = FastAPI()

@app.get("/")
async def read_item(q: Optional[str] = None):
    if q is None:
        return {"error": "No query provided"}
    
    # Load query into a Spark DataFrame
    review = ss.createDataFrame([(q,)], ["sentence"])

    # tokenize the words
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    review = tokenizer.transform(review).drop("sentence")

    # convert to bag of words
    review = cv_model.transform(review).drop("words")

    # return predict
    return {"prediction" : lr_model.transform(review).collect()[0].prediction}

@app.on_event("shutdown")
def shutdown_event():
    ss.stop()
