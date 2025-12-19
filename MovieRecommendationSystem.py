import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#we have used tmdb dataset
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

movies.head()

credits.head()

"""# **DATA PREPROCESSING**"""

movies = movies.merge(credits,on='title')

movies.head()

movies=movies[['movie_id','title','genres','overview','keywords','cast','crew']]
movies.head()

movies.isnull().sum()

movies.dropna(inplace=True)
movies.isnull().sum()

movies.duplicated().sum()

m=movies.iloc[0].genres
#it is list of dictionaries so convert into [action, adventure, fantasy, scifi]

#to convert string into list
import ast
#m=ast.literal_eval(m)

def convert(obj):
  c=[]
  for i in ast.literal_eval(obj):
    c.append(i['name'])
  return c

movies['genres']=movies['genres'].apply(convert)

movies.head()

movies['keywords']=movies['keywords'].apply(convert)

movies.head()

movies['cast'][0]
#extract 4 names of cast from cast

def convertcast(obj):
  c=[]
  counter=0
  for i in ast.literal_eval(obj):
    if counter !=4:
      c.append(i['name'])
      counter+=1
    else:
      break
  return c

movies['cast']=movies['cast'].apply(convertcast)

movies.head()

movies['crew'][0]
#extract director and name from crew

def extract(obj):
  c=[]
  for i in ast.literal_eval(obj):
    if i['job']=='Director':
      c.append(i['name'])
      break
  return c

movies['crew']=movies['crew'].apply(extract)

movies.head()

movies['overview'][0]
#it is a string convert into list

movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head()

#concatenate lists, and then merge those list into string which will output in tags

#remove spaces between names and words b/c every word will become different tag however johnny depp is one word but sue to space it will divide into 2 other tags which will create confusion for model
#for eg science fiction is one word but it will divide science into 1 tag and fiction in other

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])

movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])

movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])

movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

movies.head()

"""##**EDA**"""

# @title movie_id

from matplotlib import pyplot as plt
movies['movie_id'].plot(kind='line', figsize=(8, 4), title='movie_id')
plt.gca().spines[['top', 'right']].set_visible(False)

# @title movie_id

from matplotlib import pyplot as plt
movies['movie_id'].plot(kind='hist', bins=20, title='movie_id')
plt.gca().spines[['top', 'right',]].set_visible(False)

#no need of other columns so remove other columns and only tags is needed

newdf = movies[['movie_id','title','tags']]
newdf

newdf['tags']=newdf['tags'].apply(lambda x:" ".join(x))

newdf

newdf['tags']=newdf['tags'].apply(lambda x:x.lower())

newdf.head()

"""#**VECTORIZATION**"""

newdf['tags'][0]

newdf['tags'][1]

#convert text into vectors and calculate similarity between tags
#check closest vectors for recommendation
#bag of words -> technique to convert text into vectors
#countvectorizer to remove stop words

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(newdf['tags']).toarray()

vector.shape

vector

print(cv.get_feature_names_out())

len(cv.get_feature_names_out())

#apply stemming to your text
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

#example:
stem('learning is better')

stem(newdf['tags'][0])

newdf['tags']=newdf['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(newdf['tags']).toarray()

vector

cv.get_feature_names_out()

#calculate cosine distance since euclidean is not good for higher dimension
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(vector)

sim.shape

ls = list(enumerate(sim[0])) #similarity of first movie with all movies

sorted(ls,reverse=True,key=lambda x:x[1])[1:6]

"""##**CONTENT BASED SAMPLE CODE**"""

def recommend_content2(movie):
  movie_index = newdf[newdf['title']==movie].index[0]
  distance = sim[movie_index]
  m_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]

  for i in m_list:
    print(newdf.iloc[i[0]].title)

recommend_content2('Avatar')

recommend_content2('Batman Begins')

"""##**content based function using n recommendation**"""

def recommend_content(movie, sim, newdf, n_recommendations=10):
    movie_index = newdf[newdf['title'] == movie].index[0]
    distance = sim[movie_index]
    m_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1: n_recommendations + 1]

    recommended_movies = []
    for i in m_list:
        recommended_movies.append(newdf.iloc[i[0]].title)

    print(f"Recommended movies for '{movie}':")
    for movie in recommended_movies:
        print(movie)
    return recommended_movies

recommend_content("Avatar", sim, newdf)

"""##**POPULAR MOVIES**"""

m=pd.read_csv('tmdb_5000_movies.csv')
m
most_popular_movies = m.sort_values(by='popularity', ascending=False)
top_n = 10
top_popular_movies = most_popular_movies[['title', 'popularity', 'vote_count', 'vote_average']].head(top_n)
print("Top Popular Movies:")
print(top_popular_movies)

m=m[['id','title','genres','vote_count', 'popularity','vote_average']]
m.head()

m.isnull().sum()

m.duplicated().sum()

#to convert string into list
import ast
#m=ast.literal_eval(m)

def convert(obj):
  c=[]
  for i in ast.literal_eval(obj):
    c.append(i['name'])
  return c

m['genres']=m['genres'].apply(convert)

m

m.describe()

"""##**WEIGHTED SCORE**"""

vote_count_weight = 1.0
popularity_weight = 0.5
m['weighted_score'] = (m['vote_average'] * m['vote_count'] * vote_count_weight) + (m['popularity'] * popularity_weight)
sorted_movies = m.sort_values(by='weighted_score', ascending=False)
print(sorted_movies[['title', 'weighted_score']].head(10))

m
