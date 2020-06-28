import pandas as pd
import numpy as np
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket, json, math, sys

def fetchUserData(all_ids, outputFile=None):

    all_ids = [str(i) for i in all_ids]

    # Follow below link to know the steps in creating consumer keys
    # https://developer.twitter.com/en/apps
    # https://themepacific.com/how-to-generate-api-key-consumer-token-access-key-for-twitter-oauth/994/
    # below are my credentials kindly replace with yours
    access_token = "1091627552764915712-W8bueFC7SVmEdESK8vmnKv3HncjCxI"
    access_secret = "QAKCEnRtVRiiLbXCF64BJoOctrJcdhckJpMch5VbuQhJh"
    api_key = "5t7IyI8qhQ3ogpkY7nCty2pKj"
    api_secret = "WAQ5fpF8zZLzJNkJ8GHNVdCjW0HehEkPYhO7jRTrDDdqa85dnE"

    auth = tweepy.OAuthHandler(api_key,api_secret)
    auth.set_access_token(access_token,access_secret)
    api = tweepy.API(auth)
    api.verify_credentials()

    # # sample code to test whether the above keys are working
    # tweet = api.get_status("853359353532829696")

    user_metadata = list() # used to store all user data
    rate = 100 # max limit 100 ids at one time
    st,en,rem = 0,rate,len(all_ids)
    counter = (np.ceil(len(all_ids)/rate))

    while (rem>0):

        try:
            # print('API requests made between ids '+str(st)+' and '+str(en))
            tweets = api.statuses_lookup(id_ = all_ids[st:min(en,len(all_ids))]) # api call made here with ids in batches of 100

            rem = rem - rate
            st = en
            en = en + min(rate,rem)
            for tweet in tweets: # iterating over the obtained data
                if hasattr(tweet, 'user'):
                    data = dict()
                    data['tweetid'] = tweet._json['id']
                    data['userid'] = tweet._json['user']['id']
                    data['user_name'] = tweet._json['user']['name']
                    data['screen_name'] = tweet._json['user']['screen_name']
                    data['followers_count'] = tweet._json['user']['followers_count']
                    data['friends_count'] = tweet._json['user']['friends_count']
                    data['listed_count'] = tweet._json['user']['listed_count']
                    data['created_at'] = tweet._json['user']['created_at']
                    data['favourites_count'] = tweet._json['user']['favourites_count']
                    data['verified'] = tweet._json['user']['verified']
                    data['statuses_count'] = tweet._json['user']['statuses_count']
                    data['default_profile'] = tweet._json['user']['default_profile']
                    data['default_profile_image'] = tweet._json['user']['default_profile_image']
                    user_metadata.append(data) # all user objects are stored in user_metadata list
        except Exception as e:
            print('Error in fetching data. ERROR: {}'.format(e))


    # steps involved in creating dataframe, you can create list from above step directly but for clean code doing it separately
    user_meta_dic = dict()
    tweetid = list()
    userid = list()
    user_name = list()
    screen_name = list()
    followers_count = list()
    friends_count = list()
    listed_count = list()
    created_at = list()
    favourites_count = list()
    verified = list()
    statuses_count = list()
    default_profile = list()
    default_profile_image = list()
    for user in user_metadata:
        tweetid.append(user['tweetid'])
        userid.append(user['userid'])
        user_name.append(user['user_name'])
        screen_name.append(user['screen_name'])
        followers_count.append(user['followers_count'])
        friends_count.append(user['friends_count'])
        listed_count.append(user['listed_count'])
        created_at.append(user['created_at'])
        favourites_count.append(user['favourites_count'])
        verified.append(user['verified'])
        statuses_count.append(user['statuses_count'])
        default_profile.append(user['default_profile'])
        default_profile_image.append(user['default_profile_image'])
    user_meta_dic['tweetid'] = tweetid
    user_meta_dic['userid'] = userid
    user_meta_dic['user_name'] = user_name
    user_meta_dic['screen_name'] = screen_name
    user_meta_dic['followers_count'] = followers_count
    user_meta_dic['friends_count'] = friends_count
    user_meta_dic['listed_count'] = listed_count
    user_meta_dic['created_at'] = created_at
    user_meta_dic['favourites_count'] = favourites_count
    user_meta_dic['verified'] = verified
    user_meta_dic['statuses_count'] = statuses_count
    user_meta_dic['default_profile'] = default_profile
    user_meta_dic['default_profile_image'] = default_profile_image

    user_meta_df = pd.DataFrame(user_meta_dic, columns = ['tweetid','userid', 'user_name', 'screen_name','followers_count','friends_count','listed_count','created_at','favourites_count','verified','statuses_count','default_profile','default_profile_image'])
    if outputFile is not None:
        user_meta_df.to_csv(outputFile, encoding='utf-8-sig')


