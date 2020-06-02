package com.leseanbruneau;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.Optional;

import scala.Tuple2;

public class TestingJoins {

	public static void main(String[] args) {
		// To set Windows Environment variable
		//System.setProperty("hadoop.home.dir", "c:/hadoop");
		
		Logger.getLogger("org.apache").setLevel(Level.WARN);
		
		SparkConf conf = new SparkConf().setAppName("startingSpark").setMaster("local[*]");
		JavaSparkContext sc = new JavaSparkContext(conf);
		
		List<Tuple2<Integer, Integer>> visitsRaw = new ArrayList<>();
		visitsRaw.add(new Tuple2<>(4,18));
		visitsRaw.add(new Tuple2<>(6,4));
		visitsRaw.add(new Tuple2<>(10,9));
		
		List<Tuple2<Integer, String>> usersRaw = new ArrayList();
		usersRaw.add(new Tuple2<>(1, "John"));
		usersRaw.add(new Tuple2<>(2, "Bob"));
		usersRaw.add(new Tuple2<>(3, "Alan"));
		usersRaw.add(new Tuple2<>(4, "Doris"));
		usersRaw.add(new Tuple2<>(5, "Marybelle"));
		usersRaw.add(new Tuple2<>(6, "Raquel"));
		
		JavaPairRDD<Integer, Integer> visits = sc.parallelizePairs(visitsRaw);
		JavaPairRDD<Integer, String> users = sc.parallelizePairs(usersRaw);
		
		// Left outer join - Section 34
		//JavaPairRDD<Integer, Tuple2<Integer, Optional<String>>> joinRdd = visits.leftOuterJoin(users);
		
		// Simple left outer join to print
		//(4,(18,Optional[Doris]))
		//(6,(4,Optional[Raquel]))
		//(10,(9,Optional.empty))
		//joinRdd.collect().forEach(System.out::println);
		
		// Print names in uppercase - will throw exception if no name value from left outer join
		//joinRdd.collect().forEach(it -> System.out.println(it._2._2.get().toUpperCase()));
		
		// Print names in uppercase - will default name to BLANK if no name value from left outer join
		//joinRdd.collect().forEach(it -> System.out.println(it._2._2.orElse("blank").toUpperCase() ));
		
		// Section 35 - Right outer join - 
		//JavaPairRDD<Integer, Tuple2<Optional<Integer>, String>> joinRdd = visits.rightOuterJoin(users);
		
		// Section 35 - Print names and visits, default to zero if no value exists in right outer join
		//joinRdd.collect().forEach(it -> System.out.println(" user " + it._2._2 + " had " + it._2._1.orElse(0)));
		
		// Section 36 - Full outer join  
		JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<Integer, String>> joinRdd = visits.cartesian(users);
		
		// Section 36 - Print names and visits, default to zero if no value exists in right outer join
		joinRdd.collect().forEach(System.out::println);
		
		
		
		
		sc.close();
		

	}

}
