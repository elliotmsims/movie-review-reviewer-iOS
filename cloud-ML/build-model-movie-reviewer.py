import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: movie-reviewer <train csv> <test csv> <cv model output> <lr model output>", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="MovieReviewer").getOrCreate()
    ss = SparkSession(sc)

    print("--------- Loading training data ---------")

    def parse(x):
        try:
            return (1, x[:x.index("\",positive")])
        except:
            try:
                return (0, x[:x.index("\",negative")])
            except:
                return (None, [])

    # Load and parse the training data into (label, [word1, word2, ...])
    label_and_words = sc.textFile(sys.argv[1])\
        .map(lambda x: parse(x))\
        .filter(lambda x: x[0] is not None)

    # convert to DataFrame
    train = ss.createDataFrame(label_and_words, ["label", "sentence"])

    # tokenize the words
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    train = tokenizer.transform(train).drop("sentence")

    # convert to bag of words
    cv = CountVectorizer(inputCol="words", outputCol="features")
    cv_model = cv.fit(train)
    train = cv_model.transform(train).drop("words")
    train.cache()

    print("--------- Training model ---------")

    # train the model
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
    lr_model = lr.fit(train)

    print("--------- Loading test data ---------")

    # Load and parse the test data into (label, sentence)
    label_and_words = sc.textFile(sys.argv[2])\
        .map(lambda x: parse(x))\
        .filter(lambda x: x[0] is not None)

    # convert to DataFrame
    test = ss.createDataFrame(label_and_words, ["label", "sentence"])

    # tokenize the words
    test = tokenizer.transform(test).drop("sentence")

    # convert to bag of words
    test = cv_model.transform(test).drop("words")

    print("--------- Testing model ---------")

    # test the model
    prediction = lr_model.transform(test)
    
    # evaluate the model
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    print("Test Area Under ROC: " + str(evaluator.evaluate(prediction)))

    # save the models
    cv_model.save(sys.argv[3])
    lr_model.save(sys.argv[4])
    
    sc.stop()
    