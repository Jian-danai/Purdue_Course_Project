# Author: Bianjiang Yang
# Date: 07/10/2023
# Version: 1.0

# -------------------Part1-------------------

import pandas as pd
import math

cast = pd.read_csv("cast.csv")
titles = pd.read_csv("titles.csv")
people = pd.read_csv("people.csv")

cast.replace(r'\N', math.nan, inplace=True)
titles.replace(r'\N', math.nan, inplace=True)
people.replace(r'\N', math.nan, inplace=True)

# Save the dataframes to csv files
cast.to_csv("cast_cleaned.csv", index=False)
titles.to_csv("titles_cleaned.csv", index=False)
people.to_csv("people_cleaned.csv", index=False)

# Read cast_cleaned.csv
cast = pd.read_csv("cast_cleaned.csv")
# extract the unique values of category column and save it to a list
category = cast.category.unique().tolist()
# remove nan from the list
category.remove(category[-1])
# sort
category.sort()
# save the list to a csv file with index name as CATEGORY_ID and values as CATEGORY_NAME
category = pd.DataFrame(category, columns=['CATEGORY_NAME'])
category.index.name = 'CATEGORY_ID'
category.to_csv("category.csv")

# Read cast_cleaned.csv
cast = pd.read_csv("cast_cleaned.csv")
# get a mapping the category column with the index of category.csv
mapping = {}
for i in range(len(category)):
    mapping[category['CATEGORY_NAME'][i]] = int(category.index[i])
# replace the values of category column with the index of category.csv
cast.category = cast.category.replace(mapping)
# save the dataframe to csv file
cast.to_csv("cast_updated.csv", index=False)



# -------------------Part2-------------------
import sqlite3
import csv

con = sqlite3.connect("imdb.db")
cur = con.cursor()

with open('titles_cleaned.csv', 'r') as table:
    dr = csv.DictReader(table, delimiter = ',')
    to_db = [(i['tconst'], i['ordering'], i['title'], i['region'], i['language'],\
             i['isOriginalTitle']) for i in dr]
    cur.execute("CREATE TABLE titles ( \
        TCONST text, \
        ORDERING INTEGER, \
        TITLE text, \
        REGION text, \
        LANGUAGE text, \
        ISORIGINALTITLE REAL);")
    cur.executemany("INSERT INTO titles VALUES (?, ?, ?, ?, ?, ?);", to_db)

with open('productions.csv', 'r') as table:
    dr = csv.DictReader(table, delimiter = ',')
    to_db = [(i['tconst'], i['titleType'], i['primaryTitle'], i['originalTitle'], i['startYear'], \
            i['endYear'], i['runtimeMinutes'], i['genres']) for i in dr]
    cur.execute("CREATE TABLE productions (\
        TCONST text, \
        TITLETYPE text, \
        PRIMARYTITLE text, \
        ORIGINALTITLE text,\
        STARTYEAR INTEGER,\
        ENDYEAR INTEGER,\
        RUNTIMEMINUTES INTEGER,\
        GENRES text);")
    cur.executemany("INSERT INTO productions VALUES (?, ?, ?, ?, ?, ?, ?, ?);", to_db)

with open('ratings.csv', 'r') as table:
    dr = csv.DictReader(table, delimiter = ',')
    to_db = [(i['tconst'], i['averageRating'], i['numVotes']) for i in dr]
    cur.execute("CREATE TABLE ratings (\
        TCONST text,\
        AVERAGERATING REAL,\
        NUMVOTES INTEGER);")
    cur.executemany("INSERT INTO ratings VALUES (?, ?, ?);", to_db)

with open('people_cleaned.csv', 'r') as table:
    dr = csv.DictReader(table, delimiter = ',')
    to_db = [(i['nconst'], i['primaryName'], i['birthYear'], i['deathYear'], i['primaryProfession'],\
             i['knownForTitles']) for i in dr]
    cur.execute("CREATE TABLE people (\
        NCONST text,\
        PRIMARYNAME text,\
        BIRTHYEAR INTEGER,\
        DEATHYEAR INTEGER,\
        PRIMARYPROFESSION text,\
        KNOWNFORTITLES text);")
    cur.executemany("INSERT INTO people VALUES (?, ?, ?, ?, ?, ?);", to_db)

with open('cast_updated.csv', 'r') as table:
    dr = csv.DictReader(table, delimiter = ',')
    to_db = [(i['tconst'], i['ordering'], i['nconst'], i['category'], i['job'], i['characters']) for i in dr]
    cur.execute("CREATE TABLE cast (\
        TCONST text,\
        ORDERING INTEGER,\
        NCONST text,\
        CATEGORY text,\
        JOB text,\
        CHARACTERS text);")
    cur.executemany("INSERT INTO cast VALUES (?, ?, ?, ?, ?, ?);", to_db)

con.commit()
