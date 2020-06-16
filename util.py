import praw
from config import REDDITS, REDDITP



def authorize():
    reddit = praw.Reddit(client_id='epXADAPDLvnumg',
                         client_secret=REDDITS,
                         password=REDDITP,
                         user_agent='testscript by /u/barber5',
                         username='barber5')
    return reddit
