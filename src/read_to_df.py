import pyspark as ps

if __name__=='__main__':
    spark = ps.sql.SparkSession.builder.master("local")\
                                    .appName("casestudy-taxi")\
                                    .getOrCreate()

    mainpath = '/Users/drewthayer/galvanize/capstone-projects/avalanche-prediction/'
    dpath = 'data/data-snotel/'
    file = 'snotel_345_bison_lake.csv'
    df = spark.read.csv(mainpath + dpath + file)
