package com.leseanbruneau;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class HousePriceFields {

	public static void main(String[] args) {
		// Run on Linux		
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder().appName("House Price Analysis").master("local[*]")
				.getOrCreate();
		
		Dataset<Row> csvData = spark.read()
				.option("header", true)
				.option("inferSchema", true)
				.csv("src/main/resources/kc_house_data.csv");
		
		//csvData.describe().show();
		
		csvData = csvData.drop("id","date","waterfront","view","condition","grade","yr_renovated","zipcode","lat","long");
		
		for (String col : csvData.columns()) {
			System.out.println("The correlation between the price and " + col + ": " + csvData.stat().corr("price", col));			
		}
		
		csvData = csvData.drop("sqft_lot","sqft_lot15","yr_built","sqft_living15");
		
		for (String col1 : csvData.columns() ) {
			for (String col2 : csvData.columns()) {
				System.out.println("The correlation between the " + col1 + " and " + col2 + ": " + csvData.stat().corr(col1,col2));
			}
		}
		
		
	}

}
