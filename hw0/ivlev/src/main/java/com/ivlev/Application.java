package com.ivlev;

import com.github.davidmoten.geo.GeoHash;
import com.github.davidmoten.geo.LatLong;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.api.java.UDF3;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.expressions.Window;

import static org.apache.spark.sql.functions.avg;
import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.corr;
import static org.apache.spark.sql.functions.count;
import static org.apache.spark.sql.functions.desc;
import static org.apache.spark.sql.functions.first;
import static org.apache.spark.sql.functions.lit;
import static org.apache.spark.sql.functions.udf;
import static org.apache.spark.sql.functions.variance;

import org.apache.spark.sql.types.DataTypes;

import java.net.URL;

public class Application {

    public static void main(String[] args) {
        URL resource = Application.class.getClassLoader().getResource("AB_NYC_2019.csv");

        SparkSession sparkSession = SparkSession
                .builder()
                .config("spark.master", "local")
                .getOrCreate();

        final Dataset<Row> dataset = sparkSession.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .option("mode", "DROPMALFORMED")
                .option("escape", "\"")
                .option("quote", "\"")
                .csv(resource.getPath());

//        1. Посчитать медиану, моду и среднее и дисперсию для каждого room_type
//        ----------------------------------------------------------------------
        dataset.groupBy("room_type")
                .agg(callUDF("percentile_approx", col("price"), lit(0.5)).as("median"))
                .show();

        dataset.groupBy("room_type")
                .agg(avg("price").as("avg"))
                .show();

        dataset.groupBy("room_type")
                .agg(variance("price").as("variance"))
                .show();

        dataset.withColumn("count", count("price").over(Window.partitionBy("room_type")))
                .withColumn("price_mode", first("price").over(
                        Window.orderBy("count").partitionBy("room_type")).as("mode"))
                .groupBy("room_type")
                .agg(first("price_mode").as("mode"))
                .show();

//        2. Найти самое дорогое и самое дешевое предложение
//        --------------------------------------------------
        dataset.orderBy("price")
                .show(1);

        dataset.orderBy(desc("price"))
                .show(1);

//        3. Посчитать корреляцию между ценой и минимальный количеством ночей, кол-вом отзывов
//        ------------------------------------------------------------------------------------
        dataset.agg(
                corr("price", "minimum_nights").as("correlation_nights"),
                corr("price", "number_of_reviews").as("correlation_reviews")
        ).show();

//        4. Нужно найти гео квадрат размером 5км на 5км с самой высокой средней стоимостью жилья
//        ---------------------------------------------------------------------------------------
        UserDefinedFunction geoHash = udf((UDF3<Double, Double, Integer, String>)
                GeoHash::encodeHash, DataTypes.StringType);

        UserDefinedFunction lat = udf((UDF1<String, Double>) hash ->
                GeoHash.decodeHash(hash).getLat(), DataTypes.DoubleType);
        UserDefinedFunction lon = udf((UDF1<String, Double>) hash ->
                GeoHash.decodeHash(hash).getLon(), DataTypes.DoubleType);

        dataset.withColumn("hash", geoHash.apply(col("latitude").cast(DataTypes.DoubleType),
                col("longitude").cast(DataTypes.DoubleType),
                lit(5))
        )
                .withColumn("price", col("price").cast(DataTypes.LongType))
                .groupBy("hash")
                .agg(avg("price").as("avg_price"))
                .orderBy(col("avg_price").desc())
                .limit(1)
                .select(lat.apply(col("hash")).as("latitude"),
                        lon.apply(col("hash")).as("longitude"),
                        col("avg_price"))
                .show();
    }
}

