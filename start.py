import datetime as dt
from util import authorize

SUBREDDIT = 'coronavirus'

KEEP_ACTIONS = {'approvecomment', 'removecomment', 'spamcomment', 'approvelink', 'removelink', 'spamlink'}
COMMENT_ACTIONS = {'approvecomment', 'removecomment', 'spamcomment'}
LINK_ACTIONS = {'approvelink', 'removelink', 'spamlink'}
APPROVE_ACTIONS = {'approvecomment''approvelink'}
REMOVE_ACTIONS = {'removecomment', 'spamcomment', 'removelink', 'spamlink'}


def get_comment_actions(reddit, limit=10000):
    comment_removal_idx = {}
    for log in reddit.subreddit(SUBREDDIT).mod.log(limit=10000):
        mod = log.mod
        if mod.name == 'AutoModerator':
            continue
        action_created = dt.datetime.fromtimestamp(log.created_utc)
        action = log.action
        if action in COMMENT_ACTIONS:
            cid = log.target_fullname.split('_')[-1]
            comment = reddit.comment(id=cid)
            comment_created = dt.datetime.fromtimestamp(comment.created_utc)
            print('https://www.reddit.com{}'.format(comment.permalink))
            if cid not in comment_removal_idx:
                if comment.author is None:
                    auth_name = "__deleted_by_user__"
                else:
                    auth_name = comment.author.name
                if hasattr(comment, 'user_reports_dismissed'):
                    reports = comment.user_reports_dismissed
                else:
                    reports = []
                comment_removal_idx[cid] = {
                    'mod_actions': [],
                    'created': comment_created,
                    'body': comment.body,
                    'edited': comment.edited,
                    'ups': comment.ups,
                    'reports': reports,
                    'author': auth_name,
                    'removal_reason': comment.removal_reason,
                    'comment_obj': comment
                }

            comment_removal_idx[cid]['mod_actions'].append({log.mod.name: (action, action_created)})
    return comment_removal_idx


def get_comment_user_report_stats(reddit):
    comment_actions = get_comment_actions(reddit)
    reason_counts_by_comment = {}
    for cid, action_meta in comment_actions.items():
        for r in action_meta['reports']:
            print(r)
            reason = r[0]
            if reason not in reason_counts_by_comment:
                reason_counts_by_comment[reason] = 0
            reason_counts_by_comment[reason] += 1
    print(reason_counts_by_comment)


if __name__ == "__main__":
    reddit = authorize()
    get_comment_user_report_stats(reddit)
