'''
This code processes extracted tweets data stored in csv files.
We used Map-Reduce technique for data transformations and feature engineering;
convert raw text data into relevant features for model development
'''

import os,sys,shutil
import numpy as np
import pandas as pd
import re,os,sys,shutil
import matplotlib.pyplot as plt
import GetOldTweets3 as got
from fetchUserData import fetchUserData
from model import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, BooleanType, StringType
from textblob import TextBlob
from pyspark.sql.functions import to_timestamp


def createDir(currdir):
	if not os.path.exists(currdir):
		os.makedirs(currdir)


# Extract tweets
def searchLink(link):
	tweetCriteria = got.manager.TweetCriteria().setQuerySearch(link)
	tweet_data = got.manager.TweetManager.getTweets(tweetCriteria)
	return tweet_data

def clean(txt):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(txt).lower()).split())

def saveTweets(tweets, tweetsDataDir):

	tweetIds = []
	tweets_df = None
	for i in range(0,len(tweets)):
		d = {'link'        :  [tweets[i][0]],
				'date'        :  [tweets[i][1].date],
				'text'        :  [tweets[i][1].text],
				'tweet_id'    :  [int(tweets[i][1].id)],
				'username'    :  [tweets[i][1].username],
				'retweets'    :  [tweets[i][1].retweets],
				'favorites'   :  [tweets[i][1].favorites],
				'mentions'    :  [tweets[i][1].mentions],
				'hashtags'    :  [tweets[i][1].hashtags],
				'geo'         :  [tweets[i][1].geo],
				'permalink'   :  [tweets[i][1].permalink],
			}

		tweetIds.extend(d['tweet_id'])
		if i == 0:
			print("came1")
			tweets_df = pd.DataFrame.from_dict(d)
		else:
			print("came2")
			df = pd.DataFrame.from_dict(d)
			tweets_df = tweets_df.append(df)

	tweets_df['text'] = tweets_df.apply(lambda x: clean(x['text']), axis=1)
	tweets_df = tweets_df[['link', 'date', 'text', 'tweet_id', 'username', 'retweets', 'favorites', 'mentions', 'hashtags', 'geo', 'permalink']]
	tweets_df.to_csv(os.path.join(tweetsDataDir, 'tmp_tweets.csv'), encoding='utf-8-sig')

	return tweetIds

def extractFeatures(newsLink):

	tempDir = 'temp{}'.format(str(np.random.randint(1000, 100000, 1)[0]))
	tweetsDataDir = os.path.join(tempDir, 'datafiles')
	usersDataDir = os.path.join(tempDir, 'usersData')
	createDir(tempDir)
	createDir(tweetsDataDir)
	createDir(usersDataDir)

	# Extract Tweets
	tweetsData 	= searchLink(newsLink)
	tweetsData 	= [(newsLink, j) for j in tweetsData]
	tweetsIds 	= saveTweets(tweetsData, tweetsDataDir)

	# Extract users features
	fetchUserData(tweetsIds, outputFile=os.path.join(usersDataDir, 'tmp_users.csv'))


	# Calculate account age
	#calculate_age = lambda x: (pd.Timestamp.now() - pd.to_datetime(x['created_at'], format='%a %b %d %H:%M:%S +%f %Y')).days
	calculate_age = lambda x: (pd.Timestamp.now())
	usersDf = pd.read_csv(os.path.join(usersDataDir, 'tmp_users.csv'))
	usersDf['account_age'] 	= usersDf.apply(calculate_age, axis=1)
	usersDf['tweetid'] 		= usersDf.apply(lambda x: int(x['tweetid']), axis=1)
	cols = ['tweetid', 'userid', 'user_name', 'screen_name', 'followers_count',
			'friends_count', 'listed_count', 'account_age', 'favourites_count',
			'verified', 'statuses_count', 'default_profile', 'default_profile_image']
	usersDf = usersDf[cols]
	usersDf.to_csv(os.path.join(usersDataDir, 'tmp_users.csv'))


	# Feature engineering

	# Start spark session
	#conf = SparkConf()
	#sc = SparkContext(conf=conf)
	#spark = SparkSession(sc)
	spark = get_spark_session()

	#### TWEETS DATA ####

	# Load tweets data
	df = spark.read.option("header", "true").option("inferSchema", "true").csv(os.path.join(tweetsDataDir, '*.csv'))
	df = df.select(['link', 'tweet_id', 'text', 'retweets'])
	df = df.withColumn("retweets", df["retweets"].cast(IntegerType()))
	df = df.withColumn("tweet_id", df["tweet_id"].cast(StringType()))
	df = df.rdd.map(tuple)

	# Filter duplicates
	df = df.map(lambda x: ((x[0], x[1]), (x[2], x[3])))
	df = df.reduceByKey(lambda x,y: x)

	# Get news link & tweeet id pairs
	news_tweet_pairs = df.map(lambda x: (x[0][1], ('news', x[0][0])))


	# 1) TWEETS TEXT DATA : extract sentiments (6 bins) and subjectivity (6 bins)

	# Only text data
	df = df.map(lambda x: (x[0][0], x[1]))

	# Function for text cleaning
	def clean(txt):
		return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", str(txt).lower()).split())
	# Function to extract sentiment and subjectivity values
	def sentiment_subjectivity(txt):
		p = TextBlob(txt)
		return (p.sentiment.polarity, p.subjectivity) # We should compress this into smaller bits
	# Function to vectorize features
	def vect(x):
		a = np.array([0, 0, 0, 0, 0, 0]).astype(int)
		a[x[0][1]] = x[1]
		b = np.array([0, 0, 0, 0, 0, 0]).astype(int)
		b[x[0][2]] = x[1]
		return (x[0][0], np.concatenate((a,b)))

	# Clean text
	df1 = df.map(lambda x: (x[0], clean(x[1][0])))
	# Get sentiments and subjectivity
	df1 = df1.map(lambda x: ((x[0], sentiment_subjectivity(x[1])), 1))
	# Convert to integer scores (sentiment 0-5 and subjectivity 0-5)
	df1 = df1.map(lambda x: ((x[0][0], int((x[0][1][0]+1)*2.5), int((x[0][1][1])*5)), 1))
	# Count the number fo tweets with the same sentiment & subjectivity
	df1 = df1.reduceByKey(lambda x,y: x+y) # Count the number of tweets with the same sentiment and subjectivity
	# Vectorize features
	df1 = df1.map(lambda x: vect(x))
	# Combine text features
	df1 = df1.reduceByKey(lambda x,y: x+y)
	df1 = df1.map(lambda x: (x[0], x[1].tolist()))
	# Convert to DF
	df1 = df1.toDF(['link', 'textFeatures'])
	#print(df1.take(1))



	# 2) OTHER FEATURE : retweets count

	df2 = df.map(lambda x: (x[0], x[1][1]))
	# Set empty retweets to be 0
	df2 = df2.map(lambda x: (x[0], (x[1] if x[1] is not None else 0)))
	# Sum all retweets
	df2 = df2.reduceByKey(lambda x,y: x+y)
	# Convert to DF
	df2 = df2.toDF(['link', 'retweets'])


	# 3) USERS DATA

	# Load user data, clean the format, then convert to RDD
	udf = spark.read.option("header", "true").csv(os.path.join(usersDataDir, '*.csv'))
	udf = udf.select(['tweetid', 'followers_count', 'friends_count', 'account_age', 'verified', 'statuses_count'])
	for i in ['followers_count', 'friends_count', 'account_age', 'statuses_count']:
		udf = udf.withColumn(i, udf[i].cast(IntegerType()))
	udf = udf.withColumn('verified', udf['verified'].cast(BooleanType()))
	udf = udf.withColumn('tweetid', udf['tweetid'].cast(StringType()))
	udf = udf.rdd.map(tuple)

	# Filter duplicates (by tweetids)
	udf = udf.map(lambda x: ((x[0]), (x[1], x[2], x[3], x[4], x[5])))
	udf = udf.reduceByKey(lambda x,y: x)

	# Link to news
	udf = udf.map(lambda x: (x[0], ('users', x[1])))
	def outerJoin(x):
		news = [i[1] for i in x[1] if (i[0] == 'news')]
		users = [j[1] for j in x[1] if (j[0] == 'users')]
		return tuple([(x[0], (i,j)) for i in news for j in users])

	udf = udf.union(news_tweet_pairs) \
				.map(lambda x: (x[0], [x[1]])) \
				.reduceByKey(lambda x,y: x + y) \
				.filter(lambda x: True if 'news' in [w for v in x[1] for w in v ] else False) \
				.flatMap(lambda x: outerJoin(x)) \
				.map(lambda x: (x[1][0], (x[1][1])))
	# Output tuples: key: <news_link>   value: <'followers_count', 'friends_count', 'account_age', 'verified', 'statuses_count'>

	# Replace empty ones with 0
	udf = udf.map(lambda x: (x[0], tuple([x[1][k] if x[1][k] is not None else 0 for k in range(len(x[1]))])))
	# Compute total number of verified accounts within each tweet
	def verified_acc(n, total):
		return n/total
	# Compute followers/following ratio
	def followers_following_ratio(f_ers, f_ing):
		return (f_ers / (1 + f_ing))
	# Compute activity level (statuses_count / age)
	def activity(status_cnt, age):
		return (status_cnt / ((1+ age)))
	# Generate features
	udf = udf.map(lambda x: (x[0], (x[1][3], followers_following_ratio(x[1][0], x[1][1]), activity(x[1][4], x[1][3]))))
	# Output tuples:  key: <news_link>  value: <'verified', 'followers_to_following_ratio', 'activity'>

	# Bin featurs
	#   Bin f'ers to f'ing ratio 	to 11 bins of 0.5 intervals: [0-0.5], [0.5,1.0], [1.0, 1.5], .. [4.5, 5.0] and [5.0 ++]
	#   Bin activity				to 11 bins of 1 cnt/day intervals: [0-1], [1-2], [2-3], .. [9-10] and [10 ++]
	#   Add 1 as the value for counts
	udf = udf.map(lambda x: ((x[0], int(x[1][0]), min(10, int(x[1][1]*2)), min(10, int(x[1][2]))), 1))
	udf = udf.reduceByKey(lambda x,y: x+y)

	# Function to vectorize features
	def vect(x, n, val):
		a = np.zeros(n).astype(int)
		a[x] = val
		return a

	# Vectorize features
	udf = udf.map(lambda x: (x[0][0], np.concatenate( (vect(x[0][2], 11, x[1]), vect(x[0][3], 11, x[1]), np.array([x[0][1]])))))
	udf = udf.reduceByKey(lambda x,y: x+y)
	udf = udf.map(lambda x: (x[0], x[1].tolist()))
	udf = udf.toDF(['link', 'userFeatures'])

	# Concatenate all features
	df = df1.join(df2, ['link'])
	df = df.join(udf, ['link'])

	# Cleanup
	shutil.rmtree(tempDir)
	# # Save out
	# df.toPandas().to_csv('temp_features.csv')

	b = df.rdd.map(tuple)
	#return [1,3,4]
	#sc.stop()
	return b.collect()


# if __name__ == '__main__':

# 	newsLink = sys.argv[1]
# 	extractFeatures(newsLink)
# 	dt = pd.read_csv('temp_features.csv')
# 	print(usersFeatures, textFeatures, retweets)
