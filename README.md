# Twitter_sentiment_analysis_and_Preprocessing(NLTK)
Twitter Sentiment analysis(preprocessing using NLTK)

Introduction:-
1. Sentiment Analysis is the process of determining whether a piece of writing is positive, negative.

Why Twitter?
1. Popular microblogging site
2. 240+ million active users
3. 500 million tweets are generated everyday
4. Twitter audience varies from common man to celebrities
5. User often discuss current affairs and share personal views.
6. Tweets are small in length and hence unambiguous
7. Political party may want to know whether people support their program or not
8. A company might want find out the reviews of its products

Problem statement
1. Given a message, decide whether the message is of positive or negative sentiment. For messages conveying both a positive and negative sentiment, whichever is the stronger sentiment should be chosen
2. Aim is to detect hate speech in Tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it.

Challenges 
1. People express opinion in complex ways
2. In opinion texts, lexical content alone can be misleading
3. Out of Vocabulary Words
4. Unstructured and also non-grammatical
5. Extensive usage of acronyms like asap, lol, idk
6. Using special characters, mentions, tags
7. Lexical variation

Setup Twitter API
1. Create Twitter account and login
2. Fill Twitter application form to get access key for verification
3. Get keys after successfully fill application form
4. We get API key, API secrete key, access token, access token secrete.

Conclusion 
1. We will obtain a polarity of sentiment and display it on our webpage with 0 and 1 ( positive and negative respectively) with the help of flask framework in python and pipeline.
2. In this project we showed the importance of preprocessing of data .
3. Accuracy has increased after preprocessing and we have better results with analysis.
