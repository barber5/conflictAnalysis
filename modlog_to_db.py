from util import authorize
import datetime as dt
import sqlite3

SUBREDDIT = 'coronavirus'
COMMENT_ACTIONS = {'approvecomment', 'removecomment', 'spamcomment'}
LINK_ACTIONS = {'approvelink', 'removelink', 'spamlink'}


def persist_comment(reddit, log, cid):
    conn = sqlite3.connect('reddit.db')
    c = conn.cursor()

    # insert mod action
    action_created = dt.datetime.fromtimestamp(log.created_utc)
    action = log.action
    aid = log.id
    mid = log.mod_id36
    c.execute('INSERT INTO mod_actions (id, mod, target, action, type, created) VALUES(?, ?, ?, ?, ?, ?)',
              (aid, mid, cid, action, 0, action_created.isoformat()))
    conn.commit()
    comment = reddit.comment(id=cid)
    print('https://www.reddit.com{}'.format(comment.permalink))
    # insert mod
    try:
        c.execute('INSERT INTO mods (id, name) VALUES(?, ?)', (mid, log.mod.name))
        conn.commit()
    except Exception as e:
        print(e)

    # insert author
    if comment.author is None:
        auth_name = "__deleted_by_user__"
        auth_id = '__deleted__'
    else:
        auth_name = comment.author.name
        auth_id = comment.author.id
    try:
        c.execute('INSERT INTO authors (id, name) VALUES(?, ?)', (auth_id, auth_name))
        conn.commit()
    except Exception as e:
        print(e)

    # insert reports
    if hasattr(comment, 'user_reports_dismissed'):
        reports = comment.user_reports_dismissed
        for report in reports:
            try:
                c.execute('INSERT INTO reports (mod_action_id, report, count) VALUES(?, ?, ?)', (aid, report[0], report[1]))
                conn.commit()
            except Exception as e:
                print(e)

    # insert comment
    comment_created = dt.datetime.fromtimestamp(comment.created_utc)
    try:
        c.execute(
            'INSERT INTO comments (id, created, body, edited, ups, author, removal_reason) VALUES(?, ?, ?, ?, ?, ?, ?)',
            (cid, comment_created.isoformat(), comment.body, comment.edited, comment.ups, auth_id,
             comment.mod_reason_title))
        conn.commit()
    except Exception as e:
        print(e)
    conn.close()


def persist_link(reddit, log, lid):
    conn = sqlite3.connect('reddit.db')
    c = conn.cursor()
    submission = reddit.submission(id=lid)
    print('https://www.reddit.com{}'.format(submission.permalink))
    action_created = dt.datetime.fromtimestamp(log.created_utc)
    action = log.action
    aid = log.id
    mid = log.mod_id36
    c.execute('INSERT INTO mod_actions (id, mod, target, action, type, created) VALUES(?, ?, ?, ?, ?, ?)',
              (aid, mid, lid, action, 1, action_created.isoformat()))
    conn.commit()

    # insert mod
    try:
        c.execute('INSERT INTO mods (id, name) VALUES(?, ?)', (mid, log.mod.name))
        conn.commit()
    except Exception as e:
        print(e)

    # insert author
    if submission.author is None:
        auth_name = "__deleted_by_user__"
        auth_id = '__deleted__'
    else:
        auth_name = submission.author.name
        auth_id = submission.author.id
    try:
        c.execute('INSERT INTO authors (id, name) VALUES(?, ?)', (auth_id, auth_name))
        conn.commit()
    except Exception as e:
        print(e)

    if hasattr(submission, 'user_reports_dismissed'):
        reports = submission.user_reports_dismissed
        for report in reports:
            try:
                c.execute('INSERT INTO reports (mod_action_id, report, count) VALUES(?, ?, ?)', (aid, report[0], report[1]))
                conn.commit()
            except Exception as e:
                print(e)
    url = submission.url
    title = submission.title
    score = submission.score
    flair = submission.link_flair_text
    submission_created = dt.datetime.fromtimestamp(submission.created_utc)
    try:
        c.execute(
            'INSERT INTO submissions (id, url, title, flair, score, created, removal_reason) VALUES(?, ?, ?, ?, ?, ?, ?)',
            (submission.id, url, title, flair, score, submission_created.isoformat(), submission.mod_reason_title))
        conn.commit()
    except Exception as e:
        print(e)
    conn.close()


def persist_modlog(reddit):
    count_exceptions = 0
    for log in reddit.subreddit(SUBREDDIT).mod.log(limit=10000000):
        if count_exceptions > 10:
            print('too many exceptions, terminating')
            return
        mod = log.mod
        if mod.name == 'AutoModerator':
            continue
        action_created = dt.datetime.fromtimestamp(log.created_utc)
        action = log.action
        if action in COMMENT_ACTIONS:
            cid = log.target_fullname.split('_')[-1]
            try:
                persist_comment(reddit, log, cid)
            except Exception as e:
                count_exceptions += 1
                print('mod action already in db')
                print(e)
        elif action in LINK_ACTIONS:
            try:
                lid = log.target_fullname.split('_')[-1]
                persist_link(reddit, log, lid)
            except Exception as e:
                print('mod action already in db')
                print(e)


if __name__ == "__main__":
    reddit_all = authorize()
    persist_modlog(reddit_all)
