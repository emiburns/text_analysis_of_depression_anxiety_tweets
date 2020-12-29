#importing libraries
import os
import tweepy
import pandas as pd

#API setup
consumer_key = "EqyY4nQjA0viNhwShoKHj2dmm"
consumer_secret = "8H4XGB3xxceNjs1csQVPYotmBdqSdp2cS2DdeHlfY0uT64JgcH"
access_token = "1301978528154415104-OO22H4vpgCzdCGtsa9Ndj6YR6bH7W3"
access_token_secret = "x7Ee9roayyteO6zIbnk3FHzUT7SnxmPk40lk1pz42nF76"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

#function to retrieve bounded tweets
count = 50000

def get_tweets(searchQuery):
    
    try:
        tweets = tweepy.Cursor(api.search, q = searchQuery, tweet_mode = "extended", 
                           result_type = "mixed", lang= "en").items(count)
        tweets_list = [[tweet.created_at, tweet.id, tweet.full_text, tweet.entities,
                    tweet.retweet_count, tweet.favorite_count] for tweet in tweets]
        tweets_df = pd.DataFrame(tweets_list, columns =["Date_Created", "Tweet_ID", "Full_Text",
                                                    "Entities", "Count_Retweets", "Count_Favorites"])
    
    except tweepy.TweepError as e:
        print('Failed on: ' + str(e))
        time.sleep(3)
    
    return tweets_df

#search query for key terms
dep_tweets = get_tweets("depression OR Depression -filter:retweets -filter:'Great Depression' -filter:links")
anx_tweets = get_tweets("anxiety OR Anxiety -filter:retweets -filter:links")

#checking for duplicate tweets
dups_1 = dep_tweets.pivot_table(index =['Tweet_ID'], aggfunc = 'size')
print(dups_1)

dups_2 = anx_tweets.pivot_table(index =['Tweet_ID'], aggfunc = 'size')
print(dups_2)

#add identifying 'MI_Type' column to each df
dep_tweets["MI_Type"] = "depression"
anx_tweets["MI_Type"] = "anxiety"

#joining data frames into single df
frame = [dep_tweets, anx_tweets]
tweets_raw = pd.concat(frame)

tweets_raw.to_csv('twitter_data_raw.csv')

#checking concat executed correctly 
tweets_raw.head()
tweets_raw.info()

#writing raw data to csv file
path = '/Users/emilyburns/Documents/Data_Science/projects/twitter_nlp/data/raw_data'
output_file = os.path.join(path,'mi_twitter_data_raw.csv')

tweets_raw.to_csv(output_file, index = False)