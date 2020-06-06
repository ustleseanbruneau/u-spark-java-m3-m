package com.leseanbruneau;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceAnalysis {

	public static void main(String[] args) {
		// Run on Linux		
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder().appName("House Price Analysis").master("local[*]")
				.getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("src/main/resources/kc_house_data.csv");
		
		//csvData.printSchema();
		
		//csvData.show();
		
//		VectorAssembler vectorAssembler = new VectorAssembler();
//		vectorAssembler.setInputCols(new String[] {"bedrooms","bathrooms","sqft_living"});
//		vectorAssembler.setOutputCol("features");
		
		VectorAssembler vectorAssembler = new VectorAssembler()
				//.setInputCols(new String[] {"bedrooms","bathrooms","sqft_living"})
				.setInputCols(new String[] {"bedrooms","bathrooms","sqft_living","sqft_lot","floors","grade"})
				.setOutputCol("features");
		
		Dataset<Row> modelInputData = vectorAssembler.transform(csvData)
				.select("price","features")
				.withColumnRenamed("price", "label");

		//modelInputData.show();
		
//		Dataset<Row> [] trainingAndTestData = modelInputData.randomSplit(new double[] {0.8, 0.2});
//		Dataset<Row> trainingData = trainingAndTestData[0];
//		Dataset<Row> testData = trainingAndTestData[1];
		
//		LinearRegressionModel model = new LinearRegression()
//				.setMaxIter(10)
//				.setRegParam(0.3)
//				.setElasticNetParam(0.8)
//				.fit(trainingData);
		
		Dataset<Row> [] dataSplits = modelInputData.randomSplit(new double[] {0.8, 0.2});
		Dataset<Row> trainingAndTestData = dataSplits[0];
		Dataset<Row> holdOutData = dataSplits[1];
		
		LinearRegression linearRegression = new LinearRegression();
		
		ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
		
		ParamMap[] paramMap = paramGridBuilder.addGrid(linearRegression.regParam(), new double[] {0.01,0.1,0.5})
				.addGrid(linearRegression.elasticNetParam(), new double[] {0,0.5,1})
				.build();
		
		TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
				.setEstimator(linearRegression)
				.setEvaluator(new RegressionEvaluator().setMetricName("r2"))
				.setEstimatorParamMaps(paramMap)
				.setTrainRatio(0.8);
		
		TrainValidationSplitModel model = trainValidationSplit.fit(trainingAndTestData);
		LinearRegressionModel lrModel = (LinearRegressionModel) model.bestModel();
				
		System.out.println("The training data r2 value is " + lrModel.summary().r2() + " and the RMSE is " + lrModel.summary().rootMeanSquaredError());
		
		//model.transform(testData).show();

		System.out.println("The test data r2 value is " + lrModel.evaluate(holdOutData).r2() + " and the RMSE is " + lrModel.evaluate(holdOutData).rootMeanSquaredError());
		
		System.out.println("coefficients " + lrModel.coefficients() + " intercept : " + lrModel.intercept());
		System.out.println("reg param: " + lrModel.getRegParam() + " elastic net param : " + lrModel.getElasticNetParam());

	}

}
