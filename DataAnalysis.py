# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 16:22:28 2021

@author: bmano
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from pathlib import Path
import collections
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from statistics import mode as md

base_path = Path(__file__).parent
resultsDir = os.path.join(base_path,"Results")
genres = ["Rock","Pop","Hip-HopR&B"]
ratingValues = ["D","N","L"]
negatives = ["", "NA","NONE","NO","NAN"]
likertResponses = ["Not likely at all", "Very Unlikely", "Somewhat Unlikely", "Somewhat Likely", "Very Likely", "I have already been revisiting this artist outside of the experiment", "I heard this artist before the beginning of the study", "nan"]
retentionResponses = ["I went out of my way (searched, added to playlist, etc.) to hear this artist's song(s) which were in the experiment","I went out of my way (searched, added to playlist, etc.) to hear songs from this artist which were not in the experiment","I have not heard this artist in the time since the experiment","I heard this artist before the beginning of the study"]
#passive/active
discoveryChoices = ["Radio","Playlists on streaming services","Recommendations from friends","Discover Weekly (or equivalent on other platforms)","Music criticsjournals","Similar to artists you like"]
newSongChoices = ["Save the song and listen again later","Add the song to a playlist","Explore the artist's other music","Forget about the song"]
newArtistChoices = ["Listen to the artist's most popular songs","Listen to an album by the artist","Listen to similar artists","Forget about the artist"]
patterns = ['/','\\','x','-','|','+']

os.environ['SPOTIPY_CLIENT_ID'] = #register application with Spotify and use client key
os.environ['SPOTIPY_CLIENT_SECRET'] = #use your Spotify secret key
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888/callback'

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())

def normalizeList(lst, max):
    return [x/max for x in lst]


def buildArtistsDict(artistTracks):
    return {artistTracks[y]['Artist'][x] : [sp.artist('spotify:artist:'+artistTracks[y]['Artist'][x])['name'],artistTracks[y]['Tracks'][x].split(';')] for y in range(3) for x in range(20)}
    

def buildUserPrefsDict(user_df,playlist_df):
    return {user_df['Email'][x] : [user_df['Genre'][x],playlist_df['single_artists'][x].split(';'),playlist_df['multiple_artists'][x].split(';')] for x in range(len(user_df['Email'].values.tolist()))}


def buildUserRatingsDict(weekly_surveys,userPlaylists):
    userRatings = {}
    for x in range(len(userPlaylists['emails'].values.tolist())):
        user = userPlaylists['emails'][x]
        userRatings[user] = [[],[],[],[],[],[]]
        for y in range(1,7):
            for z in range(10):
                userRatings[user][y-1].append((userPlaylists[str(y)][x].split(';')[z], weekly_surveys[y-1]['Rate your liking for Track '+str(z+1)][weekly_surveys[y-1][weekly_surveys[y-1]['Email Address'] == userPlaylists['emails'][x]].index[0]]))
    return userRatings
    #return {userPlaylists['emails'][x] : [(userPlaylists[str(y)][x].split(';')[z], weekly_surveys[y-1]['Rate your liking for Track '+str(z+1)][weekly_surveys[y-1][weekly_surveys[y-1]['Email Address'] == userPlaylists['emails'][x]].index[0]])] for x in range(len(userPlaylists['emails'].values.tolist())) for y in range(1,6) for z in range(10)}


#only check first 2 weekly surveys for heard artists
def buildHeardArtistsDict(weekly_surveys, artists):
    heardArtists = {}
    for x in range(len(weekly_surveys[0]['Email Address'].values.tolist())):
        email = weekly_surveys[0]['Email Address'][x]
        heardArtists[email] = []
        col = 'List any of these artists that you heard prior to this experiment.'
        #first two weeks
        for y in range(2):
            #there are heard artists
            if weekly_surveys[y][col][x] != None and str(weekly_surveys[y][col][x]).upper() not in negatives:
                #get each heard artist
                for artist in weekly_surveys[y][col][x].split(','):
                    for key in artists.keys():
                        if artists[key][0].upper() == artist.upper():
                            heardArtists[email].append(key)
    return heardArtists
 
#helper functions

#get artist for track
def getArtistForTrack(track, artistInfo):
    for key in artistInfo:
        if track in artistInfo[key][1]:
            return key
    
#get artist ratings
def getArtistRatings(userRatings):
    artistRatings = collections.defaultdict(list)
    for user in userRatings:
        for artist in userRatings[user]:
            artistRatings[artist].append(userRatings[user][artist])
    return artistRatings
    
#get user rating trends
def getUserRatingsPerArtist(ratings, heardArtists, artistInfo):
    #{email: {artist: [rating1,rating2,rating3]}}
    ratingsDict = {}
    for user in ratings:
        ratingsDict[user] = collections.defaultdict(list)
        for week in range(6):
            for track in ratings[user][week]:
                #track id after spotify:track:
                artist = getArtistForTrack(track[0][14:], artistInfo)
                if artist in heardArtists[user]:
                    continue
                rating = ratingValues.index(track[1][0])
                ratingsDict[user][artist].append(rating)
    return ratingsDict
    
#get artist ratings split by mode
def getArtistRatingsSplit(userRatings,artistInfo):
    splitRatings = {}
    for artist in artistInfo:
        splitRatings[artist] = [[],[]]
    for user in userRatings:
        for artist in userRatings[user][0]:
            splitRatings[artist][0].append(userRatings[user][0][artist])
        for artist in userRatings[user][1]:
            splitRatings[artist][1].append(userRatings[user][1][artist])
    return splitRatings

#get list of user ratings by mode
def splitUserRatingsByMode(userRatings, userPrefs):
    splitRatings = {}
    for user in userRatings:
        splitRatings[user] = [collections.defaultdict(list),collections.defaultdict(list)]
        for artist in userRatings[user]:
            if artist in userPrefs[user][1]:
                splitRatings[user][0][artist] = userRatings[user][artist]
            else:
                splitRatings[user][1][artist] = userRatings[user][artist]
    return splitRatings

def getLikelihoodRatings(finalSurvey, userPrefs, artistTracks):
    likelihoodRatings = {'Rock': [[],[]], 'Pop': [[],[]], 'Hip-HopR&B': [[],[]]}
    for user in userPrefs:
        index = finalSurvey[finalSurvey['Email Address'] == user].index[0]
        genre = finalSurvey['Which was your preferred genre?'][index]
        genreIndex = genres.index(genre)
        #order of artists same in artistTracks and finalSurvey
        for x in range(20):
            artist = artistTracks[genreIndex]['Artist'][x]
            column = finalSurvey.columns[genreIndex*20+11+x]
            mode = 0
            if artist in userPrefs[user][2]:
                mode = 1
            rating = likertResponses.index(finalSurvey[column][index])
            #do not append artists that are already being explored or artists they heard before
            if rating < 5:
                likelihoodRatings[genre][mode].append(rating)
    return likelihoodRatings

#get user likelihood ratings for each artist
def getUserLikelihoodRatings(finalSurvey, userPrefs, artistTracks):
    likelihoodRatings = {}
    countOfHeardDuring = 0
    countHeardBefore = [0,0]
    for user in userPrefs:
        likelihoodRatings[user] = [{},{}]
        index = finalSurvey[finalSurvey['Email Address'] == user].index[0]
        genre = finalSurvey['Which was your preferred genre?'][index]
        genreIndex = genres.index(genre)
        #order of artists same in artistTracks and finalSurvey
        for x in range(20):
            artist = artistTracks[genreIndex]['Artist'][x]
            column = finalSurvey.columns[genreIndex*20+11+x]
            mode = 0
            if artist in userPrefs[user][2]:
                mode = 1
            rating = likertResponses.index(finalSurvey[column][index])
            #do not append artists that are already being explored or artists they heard before
            if rating < 5:
                likelihoodRatings[user][mode][artist] = rating
            if rating == 5:
                countOfHeardDuring += 1
            if rating == 6:
                countHeardBefore[mode] += 1
    print(countOfHeardDuring)
    print(countHeardBefore)
    return likelihoodRatings

def getArtistLikelihoodRatings(finalSurvey, userPrefs, artistTracks):
    likelihoodRatings = {'Rock': {}, 'Pop': {}, 'Hip-HopR&B': {}}
    for genre in genres:
        genreIndex = genres.index(genre)
        for x in range(20):
            artist = artistTracks[genreIndex]['Artist'][x]
            likelihoodRatings[genre][artist] = [[],[]]
            column = finalSurvey.columns[genreIndex*20+11+x]
            for user in userPrefs:
                index = finalSurvey[finalSurvey['Email Address'] == user].index[0]
                mode = 0
                if artist in userPrefs[user][2]:
                    mode = 1
                rating = likertResponses.index(str(finalSurvey[column][index]))
                if rating < 5:
                    likelihoodRatings[genre][artist][mode].append(rating)        
    return likelihoodRatings

#sort subjects in descending order by the amount of new music they listen to each week
def sortUsersByMusicAmount(finalSurvey):
    df = finalSurvey.sort_values(by = 'How many hours per week do you spend listening to songs you have heard a few times or less?')
    return df

#sort subjects in descending order by the amount of new music they listen to each week
def sortUsersByNewMusicAmount(finalSurvey):
    df = finalSurvey.sort_values(by = 'How many hours per week do you listen to music?')
    return df

#process follow-up survey for users
def getUserRetentionRates(followUpSurvey, artistTracks, userPrefs):
    retentionRates = {'Rock': {}, 'Pop': {}, 'Hip-HopR&B': {}}
    for user in userPrefs:
        index = followUpSurvey[followUpSurvey['Email Address'] == user].index[0]
        genre = followUpSurvey['Which was your preferred genre?'][index]
        genreIndex = genres.index(genre)
        retentionRates[genre][user] = [{},{}]
        #order of artists same in artistTracks and finalSurvey
        for x in range(20):
            artist = artistTracks[genreIndex]['Artist'][x]
            column = followUpSurvey.columns[genreIndex*20+3+x]
            mode = 0
            if artist in userPrefs[user][2]:
                mode = 1
            retentions = followUpSurvey[column][index].split(";")
            try:
                retentionRates[genre][user][mode][artist] = [retentionResponses.index(x) for x in retentions]
            except ValueError:
                print(user)
                print(x)
    return retentionRates

#process follow-up survey for artists
def getArtistRetentionRates(followUpSurvey, artistTracks, userPrefs):
    retentionRates = {'Rock': {}, 'Pop': {}, 'Hip-HopR&B': {}}
    for genre in genres:
        genreIndex = genres.index(genre)
        for x in range(20):
            artist = artistTracks[genreIndex]['Artist'][x]
            retentionRates[genre][artist] = [[],[]]
            column = followUpSurvey.columns[genreIndex*20+3+x]
            for user in userPrefs:
                if userPrefs[user][0] != genre:
                    continue
                index = followUpSurvey[followUpSurvey['Email Address'] == user].index[0]
                mode = 0
                if artist in userPrefs[user][2]:
                    mode = 1
                retentions = followUpSurvey[column][index].split(";")
                retentionRates[genre][artist][mode].append([retentionResponses.index(x) for x in retentions])
    return retentionRates

#gather all relevant data points for building a dataset to train a model on
def buildDataset(listenerProfile):
    #songDIscOpt, artistDiscOpt, avgLikelihood, weeklyDiscHrs, discoveryActiveness -> retention (0/1)
    data = []
    #retention
    classLabels = []
    for user in listenerProfile:
        data.append([user[0],user[1],user[2],user[3],user[5]])
        # if user[4] == 0.0:
        #     classLabels.append(0)
        # else:
        #     classLabels.append(1)
        classLabels.append(user[7])
    return (np.array(data), np.array(classLabels))

#calculate artist rating for multiple song artist
def calculateMultipleSongRating(ratings):
    if 0 in ratings and 1 in ratings and 2 in ratings:
        return 1
    return md(ratings)


userPlaylists = pd.read_csv("UserPlaylists.csv")
artistExcel = pd.ExcelFile("ArtistTracks.xlsx")
artistTracks = [pd.read_excel(artistExcel,genre) for genre in genres]
weekly_surveys = [pd.read_csv(os.path.join(resultsDir,"Week"+str(x)+"Survey.csv")) for x in range(1,7)]
genrePrefs = pd.read_csv("UserPrefs.csv")
finalSurvey = pd.read_csv(os.path.join(resultsDir,"FinalSurvey.csv"))
followUpSurvey = pd.read_csv(os.path.join(resultsDir, "FollowUpSurvey.csv"))

#email: [[(song1,rating1),...(songsHeard,artistsHeard)],week2,...,week6]
ratings = buildUserRatingsDict(weekly_surveys,userPlaylists)
#artistSpotifyID: [artistName,[track1,track2,track3]]
artistInfo = buildArtistsDict(artistTracks)
#email: [genre, [singleArists], [multipleArtists]]
users = buildUserPrefsDict(genrePrefs,userPlaylists)
#email: [heardArtists]
heardArtists = buildHeardArtistsDict(weekly_surveys, artistInfo)
#heardSongs = collections.defaultdict(list)
likelihoodRatings = getLikelihoodRatings(finalSurvey,users,artistTracks)
userLikelihoodRatings = getUserLikelihoodRatings(finalSurvey,users,artistTracks)
artistLikelihoodRatings = getArtistLikelihoodRatings(finalSurvey,users,artistTracks)
sortedFinalSurvey = sortUsersByNewMusicAmount(finalSurvey)
sortedFinalSurvey2 = sortUsersByMusicAmount(finalSurvey)

#data analysis
userRatings = getUserRatingsPerArtist(ratings, heardArtists, artistInfo)
artistRatings = getArtistRatings(userRatings)
userRatingsSplit = splitUserRatingsByMode(userRatings, users)
artistRatingsSplit = getArtistRatingsSplit(userRatingsSplit, artistInfo)
userRetentionRatings = getUserRetentionRates(followUpSurvey,artistTracks,users)
artistRetentionRatings = getArtistRetentionRates(followUpSurvey,artistTracks,users)

#bar graphs of average user ratings for single-song artists by play count
lst = [[],[],[]]
lstMult = [[],[],[]]
for artist in artistRatingsSplit:
    for x in range(len(artistRatingsSplit[artist][0])):
        for y in range(3):
            lst[y].append(artistRatingsSplit[artist][0][x][y])
    for x in range(len(artistRatingsSplit[artist][1])):
        for y in range(3):
            lstMult[y].append(artistRatingsSplit[artist][1][x][y])
means = [np.mean(np.array([lst[x]])) for x in range(3)]
plt.bar(range(1,4), means)
plt.title("Average user ratings for single-song artists by play count")
plt.xlabel("Listen number")
plt.ylabel("Average rating")
plt.xticks(range(1,4),["1","2","3"])
plt.ylim(0,2)
plt.show()

#count of neutral single songs based on final rating
neuts = [0,0,0]
for artist in artistRatingsSplit:
    for x in range(len(artistRatingsSplit[artist][0])):
        if artistRatingsSplit[artist][0][x][0] == 1:
            neuts[artistRatingsSplit[artist][0][x][2]] += 1
plt.bar(range(3), neuts)
plt.title("Count of final ratings for Neutral songs")
plt.xlabel("Final rating")
plt.xticks(range(3),["Dislike","Neutral","Like"])
plt.ylabel("Number of songs")
plt.show()

#count of single songs by play count and rating
ratCounts = [[lst[x].count(y) for y in range(3)] for x in range(3)]
r = range(3)
barWidth = 0.15
plt.bar([x-barWidth-0.03 for x in r],[ratCounts[x][0] for x in range(3)], width=barWidth,edgecolor='black',label='Dislike',color='darkred')
plt.bar(r, [ratCounts[x][1] for x in range(3)], width=barWidth,edgecolor='black',label='Neutral',color='gold')
plt.bar([x+barWidth+0.03 for x in r], [ratCounts[x][2] for x in range(3)], width=barWidth,edgecolor='black',label='Like',color='#50C878')
#plt.title("Counts of song ratings for each play count")
plt.xlabel("Play count",weight='bold')
plt.ylabel("Number of songs",weight='bold')
plt.legend()
plt.xticks(range(3), ["1","2","3"])
plt.ylim(0,200)
plt.show()

#bar graphs of average ratings for artists for each mode
r = range(2)
barWidth = 0.25
lst2 = [[],[]]
for artist in artistRatingsSplit:
    for y in range(2):
        for x in range(len(artistRatingsSplit[artist][y])):
            for z in range(3):
                lst2[y].append(artistRatingsSplit[artist][y][x][z])
#lst2 = [[artistRatingsSplit[artist][x][y] for artist in artistRatingsSplit for y in range(len(artistRatingsSplit[artist]))] for x in range(2)]
plt.bar(r, [np.mean(np.array(lst2[x])) for x in r])
plt.title("Average song rating for all artists split by mode")
plt.ylabel("Average rating")
plt.xlabel("Mode")
plt.xticks(range(2),["Single-song","Multiple songs"])
plt.ylim(0,2)
plt.show()

#bar graphs of average ratings for users for each mode
lst3 = [[],[]]
modeCts = [0,0]
for user in userRatingsSplit:
    for y in range(2):
        for artist in userRatingsSplit[user][y]:
            modeCts[y] += 1
            for z in range(3):
                lst3[y].append(userRatingsSplit[user][y][artist][z])
#lst3 = [[userRatingsSplit[user][x][y] for user in userRatingsSplit for y in range(len(userRatingsSplit[user]))] for x in range(2)]
plt.bar(r, [np.mean(np.array(lst3[x])) for x in r])
plt.title("Average song rating for all users split by mode")
plt.ylabel("Average rating")
plt.xlabel("Mode")
plt.xticks(range(2),["Single-song","Multiple songs"])
plt.ylim(0,2)
plt.show()

#bar graphs of average ratings split by genre and mode
lst4 = [[[],[]],[[],[]],[[],[]]]
lst4liked = [[[],[]],[[],[]],[[],[]]]
#track number of subject-artist pairs with 1+ liked song
liked = [0,0]
disliked = [0,0]
for user in userRatingsSplit:
    genreIndex = genres.index(users[user][0])
    for y in range(2):
        for artist in userRatingsSplit[user][y]:
            if 2 in userRatingsSplit[user][y][artist]:
                liked[y] += 1
                for z in range(3):
                    lst4liked[genreIndex][y].append(userRatingsSplit[user][y][artist][z])
            if 0 in userRatingsSplit[user][y][artist]:
                disliked[y] += 1
            for z in range(3):
                lst4[genreIndex][y].append(userRatingsSplit[user][y][artist][z])

r = range(3)
plt.bar(r, [np.mean(np.array(lst4[x][0])) for x in range(3)],width=barWidth,edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r], [np.mean(np.array(lst4[x][1])) for x in range(3)],width=barWidth,edgecolor='white',label='Multiple songs',color='blue')
plt.title("User ratings split by genre and mode")
plt.ylabel("Average rating")
plt.xlabel("Genres")
plt.xticks(r,["Rock","Pop","Hip-Hop//R&B"])
plt.ylim(0,2)
plt.legend()
plt.show()
#same but only for liked artists
plt.bar(r, [np.mean(np.array(lst4liked[x][0])) for x in range(3)],width=barWidth,edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r], [np.mean(np.array(lst4liked[x][1])) for x in range(3)],width=barWidth,edgecolor='white',label='Multiple songs',color='blue')
plt.title("User ratings split by genre and mode, only liked artists")
plt.ylabel("Average rating")
plt.xlabel("Genres")
plt.xticks(r,["Rock","Pop","Hip-Hop//R&B"])
plt.ylim(0,2)
plt.legend()
plt.show()

#final survey parts

#bar graph of average likelihood ratings split by genre and mode
lst5 = [[np.mean(np.array(likelihoodRatings[genre][y])) for y in range(2)] for genre in genres]
plt.bar(r, [lst5[x][0] for x in range(3)],width=barWidth,edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r], [lst5[x][1] for x in range(3)],width=barWidth,edgecolor='white',label='Multiple songs',color='blue')
plt.title("User likelihood ratings split by genre and mode")
plt.ylabel("Average likelihood rating")
plt.xlabel("Genres")
plt.xticks(r,["Rock","Pop","Hip-Hop//R&B"])
plt.ylim(0,4)
plt.legend()
plt.show()

#repeat but for liked artists only
lst5liked = [[[],[]],[[],[]],[[],[]]]
for user in userLikelihoodRatings:
    genreIndex = genres.index(users[user][0])
    for mode in range(2):
        for artist in userLikelihoodRatings[user][mode]:
            if 2 in userRatingsSplit[user][mode][artist]:
                lst5liked[genreIndex][mode].append(userLikelihoodRatings[user][mode][artist])
plt.bar(r, [np.mean(np.array(lst5liked[x][0])) for x in range(3)],width=barWidth,edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r], [np.mean(np.array(lst5liked[x][1])) for x in range(3)],width=barWidth,edgecolor='white',label='Multiple songs',color='blue')
plt.title("User likelihood ratings split by genre and mode")
plt.ylabel("Average likelihood rating")
plt.xlabel("Genres")
plt.xticks(r,["Rock","Pop","Hip-Hop//R&B"])
plt.ylim(0,4)
plt.legend()
plt.show()


#DEBUG FINAL SURVEY TO FIND MISREPORTED HEARD ARTISTS
# for user in userRatingsSplit:
#     for y in range(2):
#         for artist in userRatingsSplit[user][y]:
#             if len(userRatingsSplit[user][y][artist]) == 0:
#                 print(user)
#                 print(artistInfo[artist][0])

#bar graph of avg likelihood split just by mode
lst6 = [[],[]]
for genre in genres:
    for x in range(2):
        for y in range(len(likelihoodRatings[genre][x])):
            lst6[x].append(likelihoodRatings[genre][x][y])
plt.bar(range(2),[np.mean(np.array(lst6[x])) for x in range(2)])
plt.title("Average likelihood by mode")
plt.xlabel("Mode")
plt.xticks(range(2),["Single", "Multiple"])
plt.ylabel("Average rating")
plt.ylim(0,4)
plt.show()

#split users, split mode, song ratings
lst7 = [[[userRatingsSplit[user][y][artist][z] for artist in userRatingsSplit[user][y] for z in range(3)] for y in range(2)] for user in userRatingsSplit]
r2 = range(39)
plt.bar(r2, [np.mean(np.array(x[0])) for x in lst7], width=barWidth, edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r2], [np.mean(np.array(y[1])) for y in lst7], width=barWidth, edgecolor='white',label='Multiple songs',color='blue')
plt.title("Song ratings by mode for each subject")
plt.xlabel("Subject")
plt.ylabel("Avg rating")
plt.ylim(0,2)
plt.legend()
plt.show()

comps = []
for x in range(len(lst7)):
    #compare ratings by mode
    if np.mean(np.array(lst7[x][0])) > np.mean(np.array(lst7[x][1])):
        comps.append(0)
    elif np.mean(np.array(lst7[x][0])) == np.mean(np.array(lst7[x][1])):
        comps.append(0.5)
    else:
        comps.append(1)
print([comps.count(x) for x in range(2)])

#split users, split mode, likelihood ratings
lst8 = [[[userLikelihoodRatings[user][y][artist] for artist in userLikelihoodRatings[user][y]] for y in range(2)] for user in userLikelihoodRatings]
plt.bar(r2, [np.mean(np.array(x[0])) for x in lst8], width=barWidth, edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r2], [np.mean(np.array(y[1])) for y in lst8], width=barWidth, edgecolor='white',label='Multiple songs',color='blue')
plt.title("Likelihood ratings by mode for each subject")
plt.xlabel("Subject")
plt.ylabel("Avg rating")
plt.ylim(0,4)
plt.legend()
plt.show()

comps2 = []
for x in range(len(lst8)):
    if np.mean(np.array(lst8[x][0])) > np.mean(np.array(lst8[x][1])):
        comps2.append(0)
    elif np.mean(np.array(lst8[x][0])) > np.mean(np.array(lst8[x][1])):
        comps2.append(0.5)
    else:
        comps2.append(1)
print([comps2.count(x) for x in range(2)])

#same thing but sorting subjects by amount of new music per week
lst8 = [[[userLikelihoodRatings[user][y][artist] for artist in userLikelihoodRatings[user][y]] for y in range(2)] for user in sortedFinalSurvey['Email Address']]
plt.bar(r2, [np.mean(np.array(x[0])) for x in lst8], width=barWidth, edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r2], [np.mean(np.array(y[1])) for y in lst8], width=barWidth, edgecolor='white',label='Multiple songs',color='blue')
plt.title("Likelihood ratings by mode for each subject (Sorted)")
plt.xlabel("Subject")
plt.ylabel("Avg rating")
plt.ylim(0,4)
plt.legend()
plt.show()

#split genres, split artists, split mode, song ratings
for genre in genres:
    lst9= []
    for artist in artistTracks[genres.index(genre)]['Artist']:
        tmp = [[],[]]
        for mode in range(2):
            for x in range(len(artistRatingsSplit[artist][mode])):
                for y in range(3):
                    tmp[mode].append(artistRatingsSplit[artist][mode][x][y])
        lst9.append(tmp)
    r = range(20)
    plt.bar(r, [np.mean(np.array(x[0])) for x in lst9], width=barWidth, edgecolor='white',label='Single-song',color='red')
    plt.bar([x+barWidth for x in r], [np.mean(np.array(y[1])) for y in lst9], width=barWidth, edgecolor='white', label='Multiple songs', color='blue')
    plt.title("Average song rating for each " + genre + " artist, split by mode")
    plt.legend()
    plt.xlabel("Artists")
    plt.ylabel("Average rating")
    plt.ylim(0,2)
    #plt.xticks(range(20), [artistInfo[artist][0] for artist in artistTracks[genres.index(genre)]['Artist']])
    plt.show()
    
    comps3 = []
    for x in range(len(lst9)):
        if np.mean(np.array(lst9[x][0])) > np.mean(np.array(lst9[x][1])):
            comps3.append(0)
        elif np.mean(np.array(lst9[x][0])) > np.mean(np.array(lst9[x][1])):
            comps3.append(0.5)
        else:
            comps3.append(1)
    print([comps3.count(x) for x in range(2)])

#split genres, split artists, split mode, likelihood ratings
for genre in genres:
    lst10 = [artistLikelihoodRatings[genre][artist] for artist in artistTracks[genres.index(genre)]['Artist']]
    plt.bar(r, [np.mean(np.array(x[0])) for x in lst10], width=barWidth, edgecolor='white',label='Single-song',color='red')
    plt.bar([x+barWidth for x in r], [np.mean(np.array(y[1])) for y in lst10], width=barWidth, edgecolor='white', label='Multiple songs', color='blue')
    plt.title("Average likelihood rating for each " + genre + " artist, split by mode")
    plt.legend()
    plt.xlabel("Artists")
    plt.ylabel("Average rating")
    plt.ylim(0,4)
    #plt.xticks(range(20), [artistInfo[artist][0] for artist in artistTracks[genres.index(genre)]['Artist']])
    plt.show()
    
    comps4 = []
    for x in range(len(lst10)):
        if np.mean(np.array(lst10[x][0])) > np.mean(np.array(lst10[x][1])):
            comps4.append(0)
        elif np.mean(np.array(lst10[x][0])) > np.mean(np.array(lst10[x][1])):
            comps4.append(0.5)
        else:
            comps4.append(1)
    print([comps4.count(x) for x in range(2)])


#compare user likelihood rating with average rating for artist, split by mode
#RMSE of each mode
#lst3 and lst6, DOESN'T WORK, NOT SAME SIZE!! change lst3 to get artists based on lst6 construction
lst11 = [[[],[]],[[],[]]]
#likelihood rating based on artist rating, split by mode
lst12 = [[[],[]],[[],[]],[[],[]]]
#retention rating for same
lst13 = [[[],[]],[[],[]],[[],[]]]
for user in userRatingsSplit:
    for mode in range(2):
        for artist in userLikelihoodRatings[user][mode]:
            if artist not in userRatingsSplit[user][mode]:
                # print(user)
                # print(artist)
                continue
            lst11[mode][0].append(np.mean(np.array(userRatingsSplit[user][mode][artist]))/2)
            lst11[mode][1].append(userLikelihoodRatings[user][mode][artist]/5)
            rating = 0
            if mode == 0:
                rating = userRatingsSplit[user][mode][artist][2]
            else:
                rating = calculateMultipleSongRating(userRatingsSplit[user][mode][artist])
            lst12[rating][mode].append(userLikelihoodRatings[user][mode][artist])
            retention = userRetentionRatings[users[user][0]][user][mode][artist]
            if 3 not in retention:
                if 2 not in retention:
                    if 0 in retention and 1 in retention:
                        lst13[rating][mode].append(3)
                    elif 0 in retention:
                        lst13[rating][mode].append(1)
                    else:
                        lst13[rating][mode].append(2)
                else:
                    lst13[rating][mode].append(0)
rms = [mean_squared_error(lst11[x][0],lst11[x][1],squared=False) for x in range(2)]
plt.bar(range(2),rms)
plt.title("RMSE for avg rating vs likelihood rating, split by mode")
plt.xlabel("Mode")
plt.xticks(range(2),["Single", "Multiple"])
plt.ylabel("RMSE")
plt.ylim(0,1)
plt.show()

#plot likelihood ratings based on artist rating, split by mode
r=range(3)
plt.bar(r,[np.mean(np.array(x[0])) for x in lst12],width=barWidth,edgecolor='black',label='Single',color='#D2042D')
plt.bar([x+barWidth+0.05 for x in r],[np.mean(np.array(x[1])) for x in lst12],width=barWidth,edgecolor='black',label='Multiple',color='turquoise')
plt.ylabel('Likelihood rating',weight='bold')
plt.xticks([x+0.15 for x in r],['Disliked','Neutral','Liked'])
plt.xlabel('Artist liking',weight='bold')
plt.legend()
plt.show()

#ret mode counts based on liking, split by mode, stacked by retention mode
retLiking = [[[x[z].count(y) for y in range(1,4)] for z in range(2)] for x in lst13]
singBot = [x[0][0] for x in retLiking]
singMid = [x[0][2] for x in retLiking]
singTop = [x[0][1] for x in retLiking]
multBot = [x[1][0] for x in retLiking]
multMid = [x[1][2] for x in retLiking]
multTop = [x[1][1] for x in retLiking]
p1 = plt.bar(r,singBot,width=barWidth, edgecolor='white',label='Single, inside',color='red')
p2 = plt.bar(r,singMid,width=barWidth, edgecolor='white',label='Single, both',color='pink',bottom=singBot)
p3 = plt.bar(r,singTop,width=barWidth, edgecolor='white',label='Single, outside',color='orange',bottom=[singBot[x] + singMid[x] for x in range(3)])
p4 = plt.bar([x+barWidth for x in r],multBot,width=barWidth, edgecolor='white',label='Multiple, inside',color='green')
p5 = plt.bar([x+barWidth for x in r],multMid,width=barWidth, edgecolor='white',label='Multiple, both',color='grey',bottom=multBot)
p6 = plt.bar([x+barWidth for x in r],multTop,width=barWidth, edgecolor='white',label='Multiple, outside',color='blue',bottom=[multBot[x] + multMid[x] for x in range(3)])
#plt.title('Retention rate based on liking, split by mode, stacked by retention mode')
plt.ylabel('Retention rate',weight='bold')
plt.xticks(r, ['Disliked','Neutral','Liked'])
plt.xlabel('Artist liking',weight='bold')
plt.legend()
#plt.ylim(0.0,0.25)
plt.show()

#count of artists by rating, split by mode
plt.bar(r, [len(x[0]) for x in lst13], width=barWidth, edgecolor='black', label='Single', color='#D2042D')
plt.bar([x+barWidth+0.05 for x in r], [len(x[1]) for x in lst13], width=barWidth, edgecolor='black', label='Multiple', color='turquoise')
plt.legend()
plt.ylabel('Number of artists',weight='bold')
plt.xlabel('Artist rating',weight='bold')
plt.xticks([x+0.15 for x in r],['Dislike','Neutral','Like'])
plt.show()


#do something with artists that have already been explored during the study?


#significance of difference between average song ratings based on mode
#all pairs
sigSongAll = stats.mannwhitneyu(lst3[0],lst3[1])
print(sigSongAll)
#only liked pairs
flatLst4Liked = [[lst4liked[x][z][y] for x in range(3) for y in range(len(lst4liked[x][z]))] for z in range(2)]
sigSongLiked = stats.ttest_ind(flatLst4Liked[0], flatLst4Liked[1], equal_var = False)
print(sigSongLiked)
wilcSongLiked = stats.mannwhitneyu(flatLst4Liked[0], flatLst4Liked[1])
print(wilcSongLiked)

#significance of difference between average likelihood ratings based on mode
#all pairs
sigLikeAll = stats.mannwhitneyu(lst6[0],lst6[1])
print(sigLikeAll)

#only liked pairs
flatLst5Liked = [[lst5liked[x][z][y] for x in range(3) for y in range(len(lst5liked[x][z]))] for z in range(2)]
sigLikeLiked = stats.ttest_ind(flatLst5Liked[0], flatLst5Liked[1], equal_var = False)
print(sigLikeLiked)
wilcLikeLiked = stats.mannwhitneyu(flatLst5Liked[0], flatLst5Liked[1])
print(wilcLikeLiked)

#significance of difference between genres?

newSongResps = sortedFinalSurvey["After hearing a new song you like, what are you most likely to do?"].values
newArtistResps = sortedFinalSurvey["After hearing a new artist you like, what are you most likely to do?"].values
discResps = sortedFinalSurvey['How do you typically discover music? (Check all that apply)'].values

#retention rate
#user retention rate split by mode (ret), retained songs from study (ret1) and songs outside study (ret2)
#split by genre in ret3 and 4, songs from and not from study, respectively
#single-song retention rate = ret5 and ret6 for songs from and not from study, respectively

#compare retention rate liked vs not-liked artists = ret7
#compare retention rate and likelihood rating = ret8

#compare discovery rate and retention rate = ret9
#compare listening rate and retention rate = ret9b
#avg song rating for each retention mode = ret10
#pct of retained artists w/1+ liked listen = ret11
#avg likelihood rating of retained artists = ret12

#retention rate split by user and mode = ret13
#retention rate split by artist and mode = ret14
#retention rate sorted by weekly discovery hours = ret15
#overall retention rate for single-song artists = ret16
#average retention rate for retainers = ret17
#retention rate songs from study for retainers = ret18
#retention rate songs from outside study for retainers = ret19

#multiple-song retention rate: ret20 for overall, ret21 and ret22 for songs from and outside, respectively

#song rating for retained songs, from study vs outside = ret23
#average song rating for retainers vs non-retainers = ret24
#retention rate split by user and mode, sorted by weekly discovery hours, converted to profile value (0 for same retention rate, 1 if single is higher, 2 if multiple is higher) = ret25
#retention profile for each user, sorted (0=none, 1=only songs from study, 2=only songs outside study, 3=songs from both) = ret26
#retention rates split by mode for only liked artists = ret27

#retention rate based on song retention survey = ret28
#retention rate based on artist retention survey = ret29
#retention rate based on combination of the two = ret30
#retention rate for each subject based on artist rating split by mode sorted by discovery hours = ret31
#detailed retention rate based on discovery survey choices = ret32
#retention rate sorted by music listening hours = ret33

ret = [[],[]]
ret1 = [[],[]]
ret2 = [[],[]]
ret3 = [[[],[]],[[],[]],[[],[]]]
ret4 = [[[],[]],[[],[]],[[],[]]]
ret5 = [[],[],[]]
ret6 = [[],[],[]]
ret7 = [[],[]]
ret8 = [[],[],[],[],[]]
ret9 = []
ret9b = []
ret10 = [[],[]]
ret11 = []
ret12 = [[],[]]
ret13 = []
ret14 = [[],[],[]]
ret15 = [[] for x in range(39)]
ret16 = [[],[],[]]
ret17 = [[],[]]
ret18 = [[],[]]
ret19 = [[],[]]
ret20 = [[],[],[]]
ret21 = [[],[],[]]
ret22 = [[],[],[]]
ret23 = [[],[]]
ret24 = [[],[]]
ret25 = [[] for x in range(39)]
ret26 = [[] for x in range(39)]
ret27 = [[],[]]
# ret28 = [[] for x in newSongChoices]
# ret29 = [[] for x in newArtistChoices]
ret30 = {}
ret31 = [[] for x in range(39)]
ret32 = [[[],[]] for x in range(6)]
ret33 = [[] for x in range(39)]
#avgSongRating,likelihoodRating,mode -> retention mode (0 = none, 1 = songs in study, 2 = songs outside study, 3 = both)
X2 = []
#binary retention
y2 = []
#quadrary retention
y3 = []
heardBefore = [0,0]
discoveryChoiceCounts = [0 for x in range(6)]
count=0
for genre in genres:
    genreIndex = genres.index(genre)
    for user in userRetentionRatings[genre]:
        tmp = []
        tmp2 = [[],[]]
        tmp4 = [[],[]]
        tmp5 = [[],[],[]]
        songRatings = [[],[]]
        profile = 0
        index = sortedFinalSurvey['Email Address'].values.tolist().index(user)
        index2 = sortedFinalSurvey2['Email Address'].values.tolist().index(user)
        ret30key = str(newSongChoices.index(newSongResps[index]))+str(newArtistChoices.index(newArtistResps[index]))
        if ret30key not in ret30.keys():
            ret30[ret30key] = [[],[]]
        for y in range(2):
            for artist in userLikelihoodRatings[user][y]:
                count+=1
                #skip artists heard before study
                if 3 in userRetentionRatings[genre][user][y][artist]:
                    #print(user + " heard " + artistInfo[artist][0])
                    #get counts of artists heard before by mode
                    heardBefore[y] += 1
                    continue
                #add average song rating, likelihood rating, and mode to 2nd dataset
                X2.append([np.mean(np.array(userRatingsSplit[user][y][artist])),userLikelihoodRatings[user][y][artist],y])
                artistRating = calculateMultipleSongRating(userRatingsSplit[user][y][artist])
                for rating in userRatingsSplit[user][y][artist]:
                    songRatings[y].append(rating)
                #all artists heard after the end of the study
                if 2 not in userRetentionRatings[genre][user][y][artist]:
                    #add retention
                    ret[y].append(1)
                    #retained artist
                    tmp.append(1)
                    #retained artist in mode y
                    tmp2[y].append(1)
                    #add "retained" as simple class label for dataset 2
                    y2.append(1)
                    #add to artist rating
                    tmp5[artistRating].append(1)
                    #add song ratings to retained list
                    for rating in userRatingsSplit[user][y][artist]:
                        ret10[1].append(rating)
                    #liked artist
                    if (y==0 and userRatingsSplit[user][y][artist][2] == 2) or (y==1 and userRatingsSplit[user][y][artist].count(2) >= 2):
                        #add outside retention for liked song
                        ret7[1].append(1)
                        #add retention for liked artist
                        ret27[y].append(1)
                    #not liked
                    else:
                        #add outside retention for non-liked song
                        ret7[0].append(1)   
                    #for single-song artists
                    if y == 0:
                        #add retention of 1, based on final liking rating of single-song
                        ret16[userRatingsSplit[user][y][artist][2]].append(1)
                    #multiple-song instead
                    else:
                        ret20[artistRating].append(1)
                    #artists with 1+ liked song
                    if 2 in userRatingsSplit[user][y][artist]:
                        #add "liked song" to retained artists list
                        ret11.append(1)
                    #artists with no liked song
                    else:
                        #add "no liked song" to retained artists
                        ret11.append(0)
                    #add retention to list for likelihood rating
                    ret8[userLikelihoodRatings[user][y][artist]].append(1)
                    #add likelihood rating to list for retained artists
                    ret12[1].append(userLikelihoodRatings[user][y][artist])
                    #heard songs from study
                    if 0 in userRetentionRatings[genre][user][y][artist]:
                        #add retention for mode y
                        ret1[y].append(1)
                        #add retention for genre and mode y
                        ret3[genreIndex][y].append(1)
                        #change profile to 1 if 0, change to 3 if 2
                        if profile == 0:
                            profile = 1
                        elif profile == 2:
                            profile = 3
                        #add song ratings
                        for rating in userRatingsSplit[user][y][artist]:
                            ret23[0].append(rating)
                        #single-songs
                        if y == 0:
                            #use final rating of song
                            ret5[userRatingsSplit[user][y][artist][2]].append(1)
                        #multiple-song
                        else:
                            #use average rating of song
                            ret21[calculateMultipleSongRating(userRatingsSplit[user][y][artist])].append(1)
                        #for artists that were retained in both ways
                        if 1 in userRetentionRatings[genre][user][y][artist]:
                            #add specific retention mode for dataset 2
                            y3.append(3)
                            #also add it for retainers version of ret1 and ret2
                            tmp4[y].append(3)
                            #and add to dict
                            ret30[ret30key][y].append(3)
                        #only retained songs from study
                        else:
                            #add specific retention mode for dataset 2
                            y3.append(1)
                            #also add it for retainers version of ret1
                            tmp4[y].append(1)
                            #and dict
                            ret30[ret30key][y].append(1)
                    #heard outside songs and not songs from study
                    else:
                        #add specific retention mode for dataset 2
                        y3.append(2)
                        #also add it for retainers version of ret2
                        tmp4[y].append(2)
                        #and dict
                        ret30[ret30key][y].append(2)
                        #technically no retention for inside mode
                        if y == 0:
                            #use final rating of song
                            ret5[userRatingsSplit[user][y][artist][2]].append(0)
                        #multiple-song
                        else:
                            #use average rating of song
                            ret21[calculateMultipleSongRating(userRatingsSplit[user][y][artist])].append(0)
                        
                    #heard outside songs (and possibly songs from study)
                    if 1 in userRetentionRatings[genre][user][y][artist]:
                        #add retention for mode y
                        ret2[y].append(1)
                        #add retention for genre and mode y
                        ret4[genreIndex][y].append(1)
                        #change profile to 2 if 0, change to 3 if 1
                        if profile == 0:
                            profile = 2
                        elif profile == 1:
                            profile = 3
                        #add song ratings
                        for rating in userRatingsSplit[user][y][artist]:
                            ret23[1].append(rating)
                        #single-songs
                        if y == 0:
                            #add retention for list for rating of final listen
                            ret6[userRatingsSplit[user][y][artist][2]].append(1)
                        #multiple-song
                        else:
                            #add retention for list for average rating of listens
                            ret22[calculateMultipleSongRating(userRatingsSplit[user][y][artist])].append(1)
                    else:
                        if y == 0:
                            #add retention for list for rating of final listen
                            ret6[userRatingsSplit[user][y][artist][2]].append(0)
                        #multiple-song
                        else:
                            #add retention for list for average rating of listens
                            ret22[calculateMultipleSongRating(userRatingsSplit[user][y][artist])].append(0)
                #did not hear artist after experiment ended
                else:
                    #add no retention
                    ret[y].append(0)
                    #still consider people who retained during experiment to be retainers
                    if 0 in userRetentionRatings[genre][user][y][artist] or 1 in userRetentionRatings[genre][user][y][artist]:
                        #add retention
                        tmp.append(1)
                        #add retention based on likelihood rating
                        ret8[userLikelihoodRatings[user][y][artist]].append(1)
                    #no retention
                    else:
                        #add no retention for user
                        tmp.append(0)
                        #add no retention for likelihood rating
                        ret8[userLikelihoodRatings[user][y][artist]].append(0)
                    #since mode analysis, heard during experiment doesn't count
                    tmp2[y].append(0)
                    #for dataset 2, no retention for either class label
                    y2.append(0)
                    y3.append(0)
                    #for retainers, no retention
                    tmp4[y].append(0)
                    #no retention of songs from or outside experiment
                    ret1[y].append(0)
                    ret2[y].append(0)
                    #same but split by genre
                    ret3[genreIndex][y].append(0)
                    ret4[genreIndex][y].append(0)
                    #add likelihood rating to list for non-retained artists
                    ret12[0].append(userLikelihoodRatings[user][y][artist])
                    #add no retention to dict
                    ret30[ret30key][y].append(0)
                    #add to artist rating
                    tmp5[artistRating].append(0)
                    #single-song mode
                    if y == 0:
                        #split by song rating for songs from and outside experiment
                        ret5[userRatingsSplit[user][y][artist][2]].append(0)
                        ret6[userRatingsSplit[user][y][artist][2]].append(0)
                        #also for overall retention rate
                        ret16[userRatingsSplit[user][y][artist][2]].append(0)
                    #multiple-song mode
                    else:
                        ret20[calculateMultipleSongRating(userRatingsSplit[user][y][artist])].append(0)
                        ret21[calculateMultipleSongRating(userRatingsSplit[user][y][artist])].append(0)
                        ret22[calculateMultipleSongRating(userRatingsSplit[user][y][artist])].append(0)
                    #liked artist
                    if (y==0 and userRatingsSplit[user][y][artist][2] == 2) or (y==1 and userRatingsSplit[user][y][artist].count(2) >= 2):
                        #no retention
                        ret7[1].append(0)
                        #add no retention for liked artist
                        ret27[y].append(0)
                    #not liked artist
                    else:
                        #no retention
                        ret7[0].append(0)
                    #add song ratings to non-retention list
                    for rating in userRatingsSplit[user][y][artist]:
                        ret10[0].append(rating)
        #correlate weekly discovery hours and retention rate
        ret9.append((finalSurvey['How many hours per week do you spend listening to songs you have heard a few times or less?'][finalSurvey[finalSurvey['Email Address'] == user].index[0]],sum(tmp)/len(tmp)))
        #same but for listening hours
        ret9b.append((finalSurvey['How many hours per week do you listen to music?'][finalSurvey[finalSurvey['Email Address'] == user].index[0]],sum(tmp)/len(tmp)))
        #add user's retention, split by mode
        ret13.append(tmp2)
        #add retention based on artist ratings
        ret31[index] = tmp5
        #section for retainers
        if sum(tmp) > 0:
            for y in range(2):
                #add average retention rate for subject
                ret17[y].append(np.mean(np.array(tmp2[y])))
                #add retention rate for songs from study
                ret18[y].append(np.mean(np.array([1 if (x == 1 or x == 3) else 0 for x in tmp4[y]])))
                #add retention rate for songs outside study
                ret19[y].append(np.mean(np.array([1 if (x == 2 or x == 3) else 0 for x in tmp4[y]])))
                for rating in songRatings[y]:
                    ret24[1].append(rating)
        #non-retainers
        else:
            for y in range(2):
                for rating in songRatings[y]:
                    ret24[0].append(rating)        
#        print(str(user) + ": " + str(index))
        #add user retention list to proper index so it can be synced with other info
        ret15[index] = tmp2
        ret33[index2] = tmp2
        means = [np.mean(np.array(x)) for x in tmp2]
        if means[0] > means[1]:
            ret25[index] = 1
        elif means[0] < means[1]:
            ret25[index] = 2
        else:
            ret25[index] = 0
        ret26[index] = profile
        #discovery survey choices
        for resp in discResps[index].split(', '):
            discoveryChoiceCounts[discoveryChoices.index(resp)] += 1
            for y in range(2):
                for x in tmp4[y]:
                    ret32[discoveryChoices.index(resp)][y].append(x)
    #section for artist retention
    artistCount = 0
    for artist in artistRetentionRatings[genre]:
        tmp3 = [[],[]]
        for y in range(2):
            for rating in artistRetentionRatings[genre][artist][y]:
                #skip artists heard before study
                if 3 in rating:
                    continue
                #retained artist
                if 2 not in rating:
                    tmp3[y].append(1)
                else:
                    tmp3[y].append(0)
        artistCount += 1
        ret14[genres.index(genre)].append(tmp3)
##END RETENTION RATE SECTION

               
plt.bar(range(2), [np.mean(np.array(ret1[x])) for x in range(2)])
plt.title("Retention rate for songs from the study, split by mode")
plt.xlabel("Mode")
plt.ylabel("Retention rate")
plt.ylim(0,1)
plt.xticks(range(2), ["Single-song","Multiple-song"])
plt.show()

plt.bar(range(2), [np.mean(np.array(ret2[x])) for x in range(2)])
plt.title("Retention rate for songs from outside the study, split by mode")
plt.xlabel("Mode")
plt.ylabel("Retention rate")
plt.ylim(0,1)
plt.xticks(range(2), ["Single-song","Multiple-song"])
plt.show()

r=range(3)
plt.bar(r, [np.mean(np.array(ret3[x][0])) for x in range(3)],width=barWidth,edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r], [np.mean(np.array(ret3[x][1])) for x in range(3)],width=barWidth,edgecolor='white',label='Multiple songs',color='blue')
plt.title("Retention rates, songs in study, split by genre and mode")
plt.ylabel("Retention rate")
plt.xlabel("Genres")
plt.xticks(r,["Rock","Pop","Hip-Hop//R&B"])
plt.ylim(0,1)
plt.legend()
plt.show()

plt.bar(r, [np.mean(np.array(ret4[x][0])) for x in range(3)],width=barWidth,edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r], [np.mean(np.array(ret4[x][1])) for x in range(3)],width=barWidth,edgecolor='white',label='Multiple songs',color='blue')
plt.title("Retention rates, songs not in study, split by genre and mode")
plt.ylabel("Retention rate")
plt.xlabel("Genres")
plt.xticks(r,["Rock","Pop","Hip-Hop//R&B"])
plt.ylim(0,1)
plt.legend()
plt.show()

print("Retention rates for liked songs by mode:")
print([np.mean(np.array(x)) for x in ret7])

plt.bar(range(5), [sum(x)/len(x) for x in ret8])
#plt.title("Retention rate by likelihood rating")
plt.xlabel("Likelihood rating",weight='bold')
plt.ylabel("Retention rate",weight='bold')
plt.ylim(0.0,1.0)
#plt.xticks(range(5),["Not likely","Very unlikely","Somewhat unlikely","Somewhat likely","Very likely"])
plt.show()

plt.scatter([x[0] for x in ret9],[x[1] for x in ret9])
plt.title("Retention rate vs weekly discovery hours")
plt.xlabel("Weekly discovery hours")
plt.ylabel("Retention rate")
plt.ylim(0.0,1.0)
plt.show()

plt.scatter([x[0] for x in ret9b],[x[1] for x in ret9b])
plt.title("Retention rate vs weekly listening hours")
plt.xlabel("Weekly listening hours")
plt.ylabel("Retention rate")
plt.ylim(0.0,1.0)
plt.show()

plt.bar(range(2),[np.mean(np.array(x)) for x in ret10])
plt.title("Average song rating for retained vs not retained artists")
plt.xticks(range(2), ["Not retained","Retained"])
plt.xlabel("Retention")
plt.ylabel("Avg song rating")
plt.ylim(0.0,2.0)
plt.show()

plt.bar(r2,[np.mean(np.array(x[0])) for x in ret13],width=barWidth, edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r2],[np.mean(np.array(x[1])) for x in ret13],width=barWidth, edgecolor='white',label='Multiple-song',color='green')
plt.title("Retention rate for each user, split by mode")
plt.legend()
plt.xlabel("Subjects")
plt.ylabel("Retention rate")
plt.ylim(0.0,1.0)
plt.show()

u = [[np.mean(np.array(x[y])) for y in range(2)] for x in ret13]

r3 = range(20)
for index in range(3):
    plt.bar(r3, [np.mean(np.array(x[0])) for x in ret14[index]],width=barWidth, edgecolor='white',label='Single-song',color='red')
    plt.bar([x+barWidth for x in r3], [np.mean(np.array(x[1])) for x in ret14[index]],width=barWidth, edgecolor='white',label='Multiple-song',color='green')
    plt.title("Average retention rate for each " + genres[index] + " artist")
    plt.xlabel("Artists")
    plt.ylabel("Retention rate")
    plt.legend()
    plt.ylim(0.0,1.0)
    plt.show()

v = [[np.mean(np.array(ret14[x][y][z])) for z in range(2)] for x in range(3) for y in range(20)]

#weekly discovery hours
weeklyDiscovery = sortedFinalSurvey['How many hours per week do you spend listening to songs you have heard a few times or less?'].values
weeklyHours = sortedFinalSurvey["How many hours per week do you listen to music?"].values
print("Weekly discovery hours")
print(np.mean(weeklyDiscovery))
print(np.std(weeklyDiscovery))
plt.plot(sorted(weeklyDiscovery))
plt.title("Sorted graph of weekly discovery hours")
plt.ylabel("Hours per week")
plt.xlabel("Subjects")
plt.show()

lst8Disc = [(np.mean(np.array(lst8[x][0]+lst8[x][1])),weeklyDiscovery[x]) for x in r2]
plt.scatter([x[1] for x in lst8Disc], [x[0] for x in lst8Disc])
plt.title("Weekly new music hours vs avg likelihood rating")
plt.xlabel("Weekly discovery hours")
plt.ylabel("Avg likelihood rating")
plt.ylim(0,4)
plt.show()

#new song choice, new artist choice, avg likelihood rating, weekly discovery, retention rate, discovery activeness, profile of songs retained, profile of mode
dresps = []
for x in discResps:
    y = x.split(', ')
    #0 for passive mode, 1 for active
    dresps.append(np.mean(np.array([int(discoveryChoices.index(z)/3) for z in y])))
listenerProfile = [[newSongChoices.index(newSongResps[x]),newArtistChoices.index(newArtistResps[x]),lst8Disc[x][0],lst8Disc[x][1],sum([sum(ret15[x][y]) for y in range(2)])/sum([len(ret15[x][y]) for y in range(2)]) ,dresps[x],ret26[x],ret25[x]] for x in r2]
#avg likelihood rating and retention rate of subjects by song choice
songChoiceDict = {x : [] for x in range(len(newSongChoices))}
profileDict = collections.defaultdict(list)
for user in listenerProfile:
    songChoiceDict[user[0]].append([user[2],user[4]])
    key = str(user[0]) + str(user[1])
    profileDict[key].append(user[4])
plt.bar(range(len(newSongChoices)), [np.mean(np.array([y[0] for y in songChoiceDict[x]])) for x in range(len(newSongChoices))])
plt.title("Avg likelihood rating for subjects base on new song option")
plt.xticks(range(len(newSongChoices)),["Save song", "Add to playlist", "Listen to artist", "Forget"])
plt.ylabel("Avg likelihood")
plt.xlabel("Choices")
plt.ylim(0,4)
plt.show()

plt.bar(range(len(newSongChoices)), [np.mean(np.array([y[1] for y in songChoiceDict[x]])) for x in range(len(newSongChoices))])
plt.title("Retention rate for subjects based on new song option")
plt.xticks(range(len(newSongChoices)),["Save song", "Add to playlist", "Listen to artist", "Forget"])
plt.ylabel("Retention rate",weight='bold')
plt.xlabel("Choices",weight='bold')
plt.ylim(0.0,1.0)
plt.show()

#avg likelihood rating and retention rate of subjects by artist choice
artistChoiceDict = {x : [] for x in range(len(newArtistChoices))}
for user in listenerProfile:
    artistChoiceDict[user[1]].append([user[2],user[4]])
plt.bar(range(len(newArtistChoices)), [np.mean(np.array([y[0] for y in artistChoiceDict[x]])) for x in range(len(newArtistChoices))])
plt.title("Avg likelihood rating for subjects based on new artist option")
plt.xticks(range(len(newArtistChoices)),["Popular songs", "Album", "Similar artists", "Forget"])
plt.ylabel("Avg likelihood")
plt.xlabel("Choices")
plt.ylim(0,4)
plt.show()

plt.bar(range(len(newArtistChoices)), [np.mean(np.array([y[1] for y in artistChoiceDict[x]])) for x in range(len(newArtistChoices))])
plt.title("Retention rate for subjects based on new artist option")
plt.xticks(range(len(newArtistChoices)),["Popular songs", "Album", "Similar artists", "Forget"])
plt.ylabel("Retention rate",weight='bold')
plt.xlabel("Choices",weight='bold')
plt.ylim(0.0,1.0)
plt.show()

for x in profileDict:
    print(x + ": " + str(len(profileDict[x])) + ", " + str(np.mean(np.array(profileDict[x]))))

#retention rate vs artist liking, split by mode, stacked by retention mode
diffsSing = [abs(np.mean(np.array(ret5[x])) + np.mean(np.array(ret6[x])) - np.mean(np.array(ret16[x]))) for x in range(3)]
diffsMult = [abs(np.mean(np.array(ret21[x])) + np.mean(np.array(ret22[x])) - np.mean(np.array(ret20[x]))) for x in range(3)]
singBot = [np.mean(np.array(ret5[x])) - diffsSing[x] for x in range(3)]
singTop = [np.mean(np.array(ret6[x])) - diffsSing[x] for x in range(3)]
multBot = [np.mean(np.array(ret21[x])) - diffsMult[x] for x in range(3)]
multTop = [np.mean(np.array(ret22[x])) - diffsMult[x] for x in range(3)]
p1 = plt.bar(r,singBot,width=barWidth, edgecolor='black',label='Single, inside',color='#D2042D',hatch=patterns[0])
p2 = plt.bar(r,diffsSing,width=barWidth, edgecolor='black',label='Single, both',color='darkred',bottom=singBot,hatch=patterns[2])
p3 = plt.bar(r,singTop,width=barWidth, edgecolor='black',label='Single, outside',color='coral',bottom=[singBot[x] + diffsSing[x] for x in range(3)],hatch=patterns[1])
p4 = plt.bar([x+barWidth+0.05 for x in r],multBot,width=barWidth, edgecolor='black',label='Multiple, inside',color='turquoise',hatch=patterns[3])
p5 = plt.bar([x+barWidth+0.05 for x in r],diffsMult,width=barWidth, edgecolor='black',label='Multiple, both',color='silver',bottom=multBot,hatch=patterns[5])
p6 = plt.bar([x+barWidth+0.05 for x in r],multTop,width=barWidth, edgecolor='black',label='Multiple, outside',color='#B6D0E2',bottom=[multBot[x] + diffsMult[x] for x in range(3)],hatch=patterns[4])
#plt.title('Retention rate based on liking, split by mode, stacked by retention mode')
plt.ylabel('Retention rate',weight='bold')
plt.xticks([x+0.15 for x in r], ['Disliked','Neutral','Liked'])
plt.xlabel('Artist liking',weight='bold')
plt.legend()
plt.ylim(0.0,0.25)
plt.show()


#more analysis on listener profile
prof1 = [[x for x in listenerProfile if x[6] == y] for y in range(4)]
prof2 = [[x for x in listenerProfile if x[7] == y] for y in range(3)]

retLike = [[x[2] for x in listenerProfile if x[4] == 0.0], [x[2] for x in listenerProfile if x[4] > 0.0]]
print([np.mean(np.array(x)) for x in retLike])
stats.ttest_ind(retLike[0],retLike[1],equal_var=False)

#combination of song and artist survey responses, retention rate details
keys = sorted(ret30.keys())
numSubs = [8, 2, 1, 13, 3, 3, 3, 6]
keyNameDict = {'00':'Save/Popular','01':'Save/Album','03':'Save/Forget','10':'Playlist/Popular','11':'Playlist/Album','12':'Playlist/Similar','20':'Artist/Popular','21':'Artist/Album'}
singBot = [ret30[x][0].count(1)/len(ret30[x][0]) for x in keys]
singMid = [ret30[x][0].count(3)/len(ret30[x][0]) for x in keys]
singTop = [ret30[x][0].count(2)/len(ret30[x][0]) for x in keys]
multBot = [ret30[x][1].count(1)/len(ret30[x][1]) for x in keys]
multMid = [ret30[x][1].count(3)/len(ret30[x][1]) for x in keys]
multTop = [ret30[x][1].count(2)/len(ret30[x][1]) for x in keys]
r8 = range(8)
p1 = plt.bar(r8,singBot,width=barWidth, edgecolor='black',label='Single, inside',color='#D2042D',hatch=patterns[0])
p2 = plt.bar(r8,singMid,width=barWidth, edgecolor='black',label='Single, both',color='darkred',bottom=singBot,hatch=patterns[2])
p3 = plt.bar(r8,singTop,width=barWidth, edgecolor='black',label='Single, outside',color='coral',bottom=[singBot[x] + singMid[x] for x in range(8)],hatch=patterns[1])
p4 = plt.bar([x+barWidth+0.05 for x in r8],multBot,width=barWidth, edgecolor='black',label='Multiple, inside',color='turquoise',hatch=patterns[3])
p5 = plt.bar([x+barWidth+0.05 for x in r8],multMid,width=barWidth, edgecolor='black',label='Multiple, both',color='silver',bottom=multBot,hatch=patterns[5])
p6 = plt.bar([x+barWidth+0.05 for x in r8],multTop,width=barWidth, edgecolor='black',label='Multiple, outside',color='#B6D0E2',bottom=[multBot[x] + multMid[x] for x in range(8)],hatch=patterns[4])
plt.xlabel('Survey responses (song/artist)',weight='bold')
plt.ylabel('Retention rate',weight='bold')
plt.xticks([x+0.15 for x in r8],[keyNameDict[x] for x in keys])
plt.legend()
plt.ylim(0.0,0.6)
ax = plt.gca()
ax.tick_params(axis='x', labelrotation=45)
plt.show()

#normalized
singBot = [ret30[keys[x]][0].count(1)/len(ret30[keys[x]][0]) * numSubs[x] / 39 for x in r8]
singMid = [ret30[keys[x]][0].count(3)/len(ret30[keys[x]][0]) * numSubs[x] / 39 for x in r8]
singTop = [ret30[keys[x]][0].count(2)/len(ret30[keys[x]][0]) * numSubs[x] / 39 for x in r8]
multBot = [ret30[keys[x]][1].count(1)/len(ret30[keys[x]][1]) * numSubs[x] / 39 for x in r8]
multMid = [ret30[keys[x]][1].count(3)/len(ret30[keys[x]][1]) * numSubs[x] / 39 for x in r8]
multTop = [ret30[keys[x]][1].count(2)/len(ret30[keys[x]][1]) * numSubs[x] / 39 for x in r8]
p1 = plt.bar(r8,singBot,width=barWidth, edgecolor='black',label='Single, inside',color='#D2042D',hatch=patterns[0])
p2 = plt.bar(r8,singMid,width=barWidth, edgecolor='black',label='Single, both',color='darkred',bottom=singBot,hatch=patterns[2])
p3 = plt.bar(r8,singTop,width=barWidth, edgecolor='black',label='Single, outside',color='coral',bottom=[singBot[x] + singMid[x] for x in range(8)],hatch=patterns[1])
p4 = plt.bar([x+barWidth+0.05 for x in r8],multBot,width=barWidth, edgecolor='black',label='Multiple, inside',color='turquoise',hatch=patterns[3])
p5 = plt.bar([x+barWidth+0.05 for x in r8],multMid,width=barWidth, edgecolor='black',label='Multiple, both',color='silver',bottom=multBot,hatch=patterns[5])
p6 = plt.bar([x+barWidth+0.05 for x in r8],multTop,width=barWidth, edgecolor='black',label='Multiple, outside',color='#B6D0E2',bottom=[multBot[x] + multMid[x] for x in range(8)],hatch=patterns[4])
plt.xlabel('Survey responses (song/artist)',weight='bold')
plt.ylabel('Retention rate (Normalized)',weight='bold')
plt.xticks([x+0.15 for x in r8],[keyNameDict[x] for x in keys])
plt.legend()
plt.ylim(0.0,0.05)
ax = plt.gca()
ax.tick_params(axis='x', labelrotation=45)
plt.show()
    
#discovery choices, detailed retention rates
r6 = range(6)
singBot = [ret32[x][0].count(1)/len(ret32[x][0]) for x in r6]
singMid = [ret32[x][0].count(3)/len(ret32[x][0]) for x in r6]
singTop = [ret32[x][0].count(2)/len(ret32[x][0]) for x in r6]
multBot = [ret32[x][1].count(1)/len(ret32[x][1]) for x in r6]
multMid = [ret32[x][1].count(3)/len(ret32[x][1]) for x in r6]
multTop = [ret32[x][1].count(2)/len(ret32[x][1]) for x in r6]
p1 = plt.bar(r6,singBot,width=barWidth, edgecolor='black',label='Single, inside',color='#D2042D',hatch=patterns[0])
p2 = plt.bar(r6,singMid,width=barWidth, edgecolor='black',label='Single, both',color='darkred',bottom=singBot,hatch=patterns[2])
p3 = plt.bar(r6,singTop,width=barWidth, edgecolor='black',label='Single, outside',color='coral',bottom=[singBot[x] + singMid[x] for x in range(6)],hatch=patterns[1])
p4 = plt.bar([x+barWidth+0.05 for x in r6],multBot,width=barWidth, edgecolor='black',label='Multiple, inside',color='turquoise',hatch=patterns[3])
p5 = plt.bar([x+barWidth+0.05 for x in r6],multMid,width=barWidth, edgecolor='black',label='Multiple, both',color='silver',bottom=multBot,hatch=patterns[5])
p6 = plt.bar([x+barWidth+0.05 for x in r6],multTop,width=barWidth, edgecolor='black',label='Multiple, outside',color='#B6D0E2',bottom=[multBot[x] + multMid[x] for x in range(6)],hatch=patterns[4])
plt.xlabel('Discovery choices',weight='bold')
plt.ylabel('Retention rate (Normalized)',weight='bold')
plt.xticks([x+0.15 for x in r6],['Radio','Discover Weekly','Playlists','Friends','Critics/Journals','Similar to artists'])
plt.legend()
plt.ylim(0.0,0.3)
ax = plt.gca()
ax.tick_params(axis='x', labelrotation=45)
plt.show()

friends = [1 if x > 0 else 0 for y in range(2) for x in ret32[3][y]]

radio = [[1 if x > 0 else 0 for x in ret32[0][y]] for y in range(2)]
stats.ttest_ind(radio[0],radio[1],equal_var=False)

#weekly discovery hours vs retention rate split by mode
plt.bar(r2,[np.mean(np.array(x[0])) for x in ret15],width=barWidth, edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r2],[np.mean(np.array(x[1])) for x in ret15],width=barWidth, edgecolor='white',label='Multiple-song',color='green')
plt.title("Retention rate for each user, split by mode")
plt.legend()
plt.xlabel("Subjects")
plt.ylabel("Retention rate")
plt.ylim(0.0,1.0)
plt.show()

#weekly listening hours vs retention rate, split by mode
plt.bar(r2,[np.mean(np.array(x[0])) for x in ret33],width=barWidth, edgecolor='white',label='Single-song',color='red')
plt.bar([x+barWidth for x in r2],[np.mean(np.array(x[1])) for x in ret33],width=barWidth, edgecolor='white',label='Multiple-song',color='green')
plt.title("Retention rate for each user, split by mode")
plt.legend()
plt.xlabel("Subjects")
plt.ylabel("Retention rate")
plt.ylim(0.0,1.0)
plt.show()

#weekly discovery hours vs retention rate split by artist rating
plt.bar([x-barWidth for x in r2],[np.mean(np.array(x[0])) for x in ret31],width=barWidth, edgecolor='white',label='Disliked',color='red')
plt.bar(r2,[np.mean(np.array(x[1])) for x in ret31],width=barWidth, edgecolor='white',label='Neutral',color='green')
plt.bar([x+barWidth for x in r2],[np.mean(np.array(x[2])) for x in ret31],width=barWidth, edgecolor='white',label='Liked',color='blue')
plt.title("Retention rate for each user, split by artist rating, sorted by discovery hours")
plt.legend()
plt.xlabel("Subjects")
plt.ylabel("Retention rate")
plt.ylim(0.0,1.0)
plt.show()

#plot density of likelihood ratings and retention rates
plt.scatter([np.mean(np.array(x[0])) for x in ret13],[0 for x in range(39)])
plt.show()

plt.scatter([np.mean(np.array(x[1])) for x in ret13],[0 for x in range(39)])
plt.show()

plt.scatter([np.mean(np.array(x[0])) for x in lst8Disc],[0 for x in range(39)])
plt.show()


# #build dataset
#ADD DISCOVERY (ACTIVE/PASSIVE)
X, y = buildDataset(listenerProfile)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
k_fold = KFold(n_splits=5)
k_fold.get_n_splits(X)
cvm = LinearRegression()
score = cross_val_score(cvm,X,y,cv=k_fold,n_jobs=1)
print(score)
print(sum(score)/len(score))
#gnb.fit(X_train, y_train)
#y_pred = gnb.predict(X_test)
#print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

#second dataset with all subj-artist pairs (not heard before)
#DOESN'T WORK!! TOO MANY NON-RETAINED
# #train Naive Bayes model on avg song rating, weekly discovery hours, mode (ANY OTHERS?) to predict retention
# gnb = GaussianNB()
# kf = KFold(n_splits = 10)
# kf.get_n_splits(X2)
# score2 = cross_val_score(gnb,X2,y3,cv=kf,n_jobs=1)
# print(score2)
# print(sum(score2)/len(score2))

#statistical/significance analysis
#likelihood rating, single vs multiple
stats.ttest_ind(lst6[0],lst6[1], equal_var=False)

#likelihood rating, retained vs non-retained
stats.ttest_ind(ret12[0],ret12[1], equal_var=False)

#song rating, single vs multiple
stats.ttest_ind(lst3[0],lst3[1], equal_var=False)

#song rating, retained vs not retained
stats.ttest_ind(ret10[0],ret10[1], equal_var=False)

#song rating, retainers vs non retainers
stats.ttest_ind(ret24[0],ret24[1], equal_var=False)

#song rating, retained outside vs retained from study
stats.ttest_ind(ret23[0],ret23[1], equal_var=False)

#retention rate, single vs multiple
stats.ttest_ind(ret[0],ret[1],equal_var=False)

#ret rate, songs from study, single vs multiple
stats.ttest_ind(ret1[0],ret1[1],equal_var=False)

#ret rate, songs outside study, single vs multiple
stats.ttest_ind(ret2[0],ret2[1],equal_var=False)

#retention rate, not-liked artists vs liked artists
stats.ttest_ind(ret7[0],ret7[1],equal_var=False)

#retention rate, liked artists, single vs multiple
stats.ttest_ind(ret27[0],ret27[1],equal_var=False)