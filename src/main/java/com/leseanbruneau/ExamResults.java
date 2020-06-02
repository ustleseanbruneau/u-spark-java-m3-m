package com.leseanbruneau;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.*;

public class ExamResults {

	public static void main(String[] args) {
		// To set Windows Environment variable
		//System.setProperty("hadoop.home.dir", "c:/hadoop");		
		//Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		// Note: Windows need to add .config() for directory, windows directory does not have to exist to run
		//SparkSession spark = SparkSession.builder().appName("testingSql").master("local[*]")
		//		.config("spark.sql.warehouse.dir","file:///c:/temp/spark")
		//		.getOrCreate();
		
		
		
		// Run on Linux		
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		SparkSession spark = SparkSession.builder().appName("testingSql").master("local[*]")
				.getOrCreate();
		
		// Chapter 72 - More Aggregations
		//Dataset<Row> dataset = spark.read().option("header", true).option("inferSchema", true).csv("src/main/resources/exams/students.csv");
		//Dataset<Row> dataset = spark.read().option("header", true).csv("src/main/resources/exams/students.csv");
		
		//Column score = dataset.col("score");
		
		//dataset = dataset.groupBy("subject").count();
		//dataset = dataset.groupBy("subject").max("score");
		
		// one aggregate column
		//dataset = dataset.groupBy("subject").agg(max(col("score").cast(DataTypes.IntegerType) ).alias("max score"));
		
		// multiple aggregate columns
//		dataset = dataset.groupBy("subject").agg(max( col("score") ).alias("max score"), 
//				min( col("score") ).alias("min score"),
//				avg( col("score") ).alias("avg score") );
		
		// chapter 73 - Practical Exercise
//		dataset = dataset.groupBy("subject").pivot("year").agg( round( avg(col("score")), 2 ).alias("average"), 
//				round( stddev(col("score")),2 ).alias("stddev"));
		
		// Chapter 74 - User Defined Functions 
		
		// new / modern way of syntax with Lambda function
		// one column filter
//		spark.udf().register("hasPassed", (String grade) -> { 
//			
//			return grade.startsWith("A") || grade.startsWith("B") || grade.startsWith("C"); 
//			
//			}, DataTypes.BooleanType );
		
		spark.udf().register("hasPassed", (String grade, String subject) -> { 
			
			if (subject.contentEquals("Biology")) {
				if (grade.startsWith("A")) return true;
				return false;
			}
			
			return grade.startsWith("A") || grade.startsWith("B") || grade.startsWith("C"); 
			
			}, DataTypes.BooleanType );
		
		Dataset<Row> dataset = spark.read().option("header", true).csv("src/main/resources/exams/students.csv");
		
		//dataset = dataset.withColumn("pass", lit( col("grade").equalTo("A+") ));
		dataset = dataset.withColumn("pass", callUDF("hasPassed", col("grade"), col("subject")) );
		
		dataset.show();

	}

}
