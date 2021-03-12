import pickle
import random
import sqlite3
import datetime as dt
from collections import Counter
from math import log
import plotly.graph_objects as go
from dateutil import parser
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder

import sage
from config import APPROVE_ACTIONS, REMOVE_ACTIONS
from util import authorize
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

COMMON_ACTION_THRESHOLD = 100
IGNORE_MODS = {'BotTerminator', 'CovidPoliticsBot'}
reddit_all = authorize()


def get_all_comments_from_db():
    conn = sqlite3.connect('reddit.db')
    c = conn.cursor()
    columns = ['maid', 'mid', 'cid', 'action', False, 'action_created', False, 'comment_created', 'body', 'edited',
               'ups', 'aid', 'removal_reason', False, 'mod_name', False, 'author', False, 'report', 'count']
    cid_idx = {}
    print('getting comments from db')
    rows = c.execute(
        'SELECT * FROM mod_actions AS ma JOIN comments AS c ON c.id=ma.target JOIN mods AS m ON m.id=ma.mod '
        'JOIN authors AS a ON c.author=a.id LEFT JOIN reports AS r ON ma.id=r.mod_action_id').fetchall()
    print('fetched db result')
    for i, row in enumerate(rows):
        if i % 1000 == 0:
            print('processing row {}'.format(i))
        next_obj = {}
        for j, heading in enumerate(columns):
            next_obj[heading] = row[j]
        cid = next_obj['cid']
        mod = next_obj['mod_name']
        if mod in IGNORE_MODS:
            continue
        if cid not in cid_idx:
            next_obj['reports'] = {}
            next_obj['actions'] = {}
            cid_idx[cid] = next_obj
        if next_obj['count'] is None:
            cid_idx[cid]['reports']['No report associated'] = {
                'report': 'No report associated',
                'count': 1
            }
        elif next_obj['report'] is None:
            cid_idx[cid]['reports']['null'] = {
                'report': 'null',
                'count': next_obj['count']
            }
        else:
            cid_idx[cid]['reports'][next_obj['report']] = {
                'report': next_obj['report'],
                'count': next_obj['count']
            }
        cid_idx[cid]['actions'][next_obj['maid']] = {
            'action': next_obj['action'],
            'maid': next_obj['maid'],
            'moderator': next_obj['mod_name'],
            'action_created': next_obj['action_created']
        }

    return list(cid_idx.values())


def get_codebook():
    with open('codebook.pkl', 'rb') as fi:
        auto_map = pickle.load(fi)
    result = {}
    for category, reports in auto_map.items():
        for report in reports:
            if report in result:
                print('duplicate in codebook, report: {}, existing: {}, current: {}'.format(report, result[report],
                                                                                            category))
            if category == 'o':
                result[report] = 'other'
            elif category == 'c':
                result[report] = 'incivility'
            elif category == 'q':
                result[report] = 'information quality'
            elif category == 'p':
                result[report] = 'politics'
            elif category == 's':
                result[report] = 'spam'
            elif category == 'n':
                result[report] = 'no report'
            elif category == 'b':
                result[report] = 'bot generated report'
    return result


def code_reports(comments):
    codebook = get_codebook()
    for comment in comments:
        for report in comment['reports'].values():
            reason = report['report']
            if reason not in codebook:
                report['report'] = 'no report associated'
            else:
                report['report'] = codebook[reason]
    return comments


def fix_none_reports(comments=None):
    if comments is None:
        comments = get_all_comments_from_db()
    for comment in comments:
        for report in comment['reports'].values():
            reason = report['report']
            if report['count'] == 'jquiz1852':
                cid = comment['cid']
                comment_loaded = reddit_all.comment(id=cid)
                print('https://reddit.com{}'.format(comment_loaded.permalink))
                print(cid)
                if hasattr(comment_loaded, 'mod_reports_dismissed'):
                    conn = sqlite3.connect('reddit.db')
                    c = conn.cursor()
                    reports = comment_loaded.mod_reports_dismissed
                    for mr in reports:
                        try:
                            for action in comment['actions'].values():
                                c.execute('INSERT INTO reports (mod_action_id, report, count) VALUES(?, ?, ?)',
                                          (action['maid'], mr[0], mr[1]))
                                conn.commit()
                        except Exception as e:
                            print(e)
                    conn.close()


def count_removal_reasons(comments=None):
    if comments is None:
        comments = get_all_comments_from_db()
    count_idx = {}
    for comment in comments:
        for report in comment['reports'].values():
            reason = report['report']
            if reason not in count_idx:
                count_idx[reason] = 0
            count_idx[reason] += 1
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Report type statistics')
    ax.set_xlabel('report reason')
    ax.set_ylabel('comment count')
    ax.bar(count_idx.keys(), count_idx.values())
    plt.show()


def get_most_recent_action(comment):
    # get most recent action
    most_recent_action = list(comment['actions'].values())[0]
    most_recent_time = list(comment['actions'].values())[0]['action_created']
    for action in comment['actions'].values():
        if action['action_created'] > most_recent_time:
            most_recent_action = action
            most_recent_time = action['action_created']
    return most_recent_action


def get_most_recent_action_by_mod(comment):
    mod_action_idx = {}
    for action in comment['actions'].values():
        mod = action['moderator']
        if mod not in mod_action_idx:
            mod_action_idx[mod] = []
        mod_action_idx[mod].append(action)
    result = {}
    for mod, actions in mod_action_idx.items():
        most_recent_action = actions[0]
        most_recent_time = actions[0]['action_created']
        for action in actions:
            if action['action_created'] > most_recent_time:
                most_recent_action = action
                most_recent_time = action['action_created']
        result[mod] = most_recent_action
    return result


def detect_conflict(comment):
    approved = False
    removed = False
    mod_action_idx = {}
    for action in comment['actions'].values():
        action_name = action['action']
        moderator = action['moderator']
        if moderator not in mod_action_idx:
            mod_action_idx[moderator] = []
        mod_action_idx[moderator].append(action)
    if len(mod_action_idx.keys()) == 1:
        return False
    for mod, actions in mod_action_idx.items():
        last_action = actions[0]['action']
        last_action_time = actions[0]['action_created']
        for action in actions:
            if action['action_created'] > last_action_time:
                last_action = action['action']
                last_action_time = action['action_created']
        if last_action in APPROVE_ACTIONS:
            approved = True
        if last_action in REMOVE_ACTIONS:
            removed = True
    return approved and removed


def get_number_of_mods(comment):
    mod_action_idx = {}
    for action in comment['actions'].values():
        action_name = action['action']
        moderator = action['moderator']
        if moderator not in mod_action_idx:
            mod_action_idx[moderator] = []
        mod_action_idx[moderator].append(action)
    return len(mod_action_idx.keys())


def detect_concordance(comment):
    approved = False
    removed = False
    mod_action_idx = {}
    for action in comment['actions'].values():
        action_name = action['action']
        moderator = action['moderator']
        if moderator not in mod_action_idx:
            mod_action_idx[moderator] = []
        mod_action_idx[moderator].append(action)
    if len(mod_action_idx.keys()) == 1:
        return False
    for mod, actions in mod_action_idx.items():
        last_action = actions[0]['action']
        last_action_time = actions[0]['action_created']
        for action in actions:
            if action['action_created'] > last_action_time:
                last_action = action['action']
                last_action_time = action['action_created']
        if last_action in APPROVE_ACTIONS:
            approved = True
        if last_action in REMOVE_ACTIONS:
            removed = True
    return not (approved and removed)


def mod_action_statistics(comments=None):
    print('getting mod action stats')
    if comments is None:
        comments = get_all_comments_from_db()
    reason_action_count_idx = {}
    for comment in comments:
        action = get_most_recent_action(comment)
        for report in comment['reports'].values():
            reason = report['report']
            if reason not in reason_action_count_idx:
                reason_action_count_idx[reason] = {
                    'approve': 0,
                    'remove': 0
                }
            action_name = action['action']
            if action_name in APPROVE_ACTIONS:
                reason_action_count_idx[reason]['approve'] += 1
            if action_name in REMOVE_ACTIONS:
                reason_action_count_idx[reason]['remove'] += 1
        if len(comment['reports']) == 0:
            if 'none' not in reason_action_count_idx:
                reason_action_count_idx['none'] = {}
            action_name = action['action']
            if action_name not in reason_action_count_idx['none']:
                reason_action_count_idx['none'][action_name] = 0
            reason_action_count_idx['none'][action_name] += 1
    df_array = []
    for reason, count_meta in reason_action_count_idx.items():
        df_array.append(count_meta)
    df = pd.DataFrame(df_array, index=reason_action_count_idx.keys())
    df.plot(kind='bar')
    plt.show()


def compare_conflict_to_non_conflict(comments=None):
    if comments is None:
        comments = get_all_comments_from_db()
    conflict_counts = {}
    non_conflict_counts = {}
    for comment in comments:
        if detect_conflict(comment):
            action = get_most_recent_action(comment)
            for report in comment['reports'].values():
                reason = report['report']
                if reason not in conflict_counts:
                    conflict_counts[reason] = 0
                conflict_counts[reason] += 1
        else:
            for report in comment['reports'].values():
                reason = report['report']
                if reason not in non_conflict_counts:
                    non_conflict_counts[reason] = 0
                non_conflict_counts[reason] += 1
    keeper_reasons = set([])
    total_nc_comments = 0
    for reason, count in non_conflict_counts.items():
        if count > COMMON_ACTION_THRESHOLD:
            keeper_reasons.add(reason)
            total_nc_comments += count
    total_c_comments = 0
    for reason, count in conflict_counts.items():
        if reason in keeper_reasons:
            total_c_comments += count
    print('reason\tnon-conflict proportion\tconflict proportion')
    for reason, count in non_conflict_counts.items():
        if reason in keeper_reasons:
            if reason in conflict_counts:
                cc = conflict_counts[reason]
            else:
                cc = 0
            print('{}\t{}\t{}'.format(reason, float(count) / total_nc_comments, float(cc) / total_c_comments))
    for reason, count in conflict_counts.items():
        if reason in keeper_reasons:
            if reason not in non_conflict_counts:
                print('{}\t{}\t{}'.format(reason, 0, float(count) / total_c_comments))


def mod_conflict_statistics(comments=None):
    # how are reasons distributed over conflicts vs non-conflicts
    # how are final actions in a conflict distributed
    # for each reason, how many conflicts vs non-conflicts
    # check as well for concordance i.e. 2 diff mods take same action
    count_meta = {}
    print('getting mod conflict stats')
    if comments is None:
        comments = get_all_comments_from_db()
    final_idx = {}  # final action taken count
    report_count_idx = {}
    for comment in comments:
        if detect_conflict(comment):
            action = get_most_recent_action(comment)
            action_name = action['action']
            if action_name not in final_idx:
                final_idx[action_name] = 0
            final_idx[action_name] += 1
            for report in comment['reports'].values():
                reason = report['report']
                if reason not in report_count_idx:
                    report_count_idx[reason] = 0
                report_count_idx[reason] += 1
            if len(comment['reports']) == 0:
                if 'none' not in report_count_idx:
                    report_count_idx['none'] = 0
                report_count_idx['none'] += 1
    print('final action on conflicted items count')
    for action, count in final_idx.items():
        print('{}\t{}'.format(action, count))
    print('conflicted item report reasons')
    for reason, count in report_count_idx.items():
        print('{}\t{}'.format(reason, count))


def last_entry(comments=None):
    if comments is None:
        comments = get_all_comments_from_db()
    print('last mod action')
    print(comments[-1])


def get_last_action_for_each_mod(comments=None):
    if comments is None:
        comments = get_all_comments_from_db()
    mod_idx = {}
    for comment in comments:
        for action in comment['actions'].values():
            mod = action['moderator']
            action_taken = action['action']
            action_created = action['action_created']
            if mod not in mod_idx:
                mod_idx[mod] = {
                    'remove': 0,
                    "approve": 0,
                    "last": action_created
                }
            if action_taken in APPROVE_ACTIONS:
                mod_idx[mod]["approve"] += 1
            if action_taken in REMOVE_ACTIONS:
                mod_idx[mod]["remove"] += 1
            if action_created > mod_idx[mod]["last"]:
                mod_idx[mod]["last"] = action_created
    print('{}\t{}\t{}\t{}'.format('mod', 'approvals', 'removals', 'last action'))
    for mod, mod_meta in mod_idx.items():
        print('{}\t{}\t{}\t{}'.format(mod, mod_meta["approve"], mod_meta["remove"], mod_meta["last"]))


def analyze_report_long_tail(comments=None):
    if comments is None:
        comments = get_all_comments_from_db()
    long_tail = get_reports_long_tail(comments)
    idx_by_total = {}
    idx_by_tuple = {}
    for reason, action, count in long_tail:
        if reason not in idx_by_total:
            idx_by_total[reason] = 0
        idx_by_total[reason] += count
        if reason not in idx_by_tuple:
            idx_by_tuple[reason] = []
        idx_by_tuple[reason].append((reason, action, count))
    for r, count in idx_by_total.items():
        print('{}\t{}'.format(r, count))


def get_reports_long_tail(comments=None):
    if comments is None:
        comments = get_all_comments_from_db()
    reason_action_count_idx = {}
    for comment in comments:
        action = get_most_recent_action(comment)
        for report in comment['reports'].values():
            reason = report['report']
            if reason not in reason_action_count_idx:
                reason_action_count_idx[reason] = {
                    'approve': 0,
                    'remove': 0
                }
            action_name = action['action']
            if action_name in APPROVE_ACTIONS:
                reason_action_count_idx[reason]['approve'] += 1
            if action_name in REMOVE_ACTIONS:
                reason_action_count_idx[reason]['remove'] += 1
        if len(comment['reports']) == 0:
            if 'none' not in reason_action_count_idx:
                reason_action_count_idx['none'] = {}
            action_name = action['action']
            if action_name not in reason_action_count_idx['none']:
                reason_action_count_idx['none'][action_name] = 0
            reason_action_count_idx['none'][action_name] += 1
    result = []
    for reason, count_meta in reason_action_count_idx.items():
        total_actions = 0
        for action, count in count_meta.items():
            total_actions += count
        if total_actions <= COMMON_ACTION_THRESHOLD:
            for action, count in count_meta.items():
                result.append((reason, action, count))
    return result


def write_clean_csv(comments):
    w = csv.writer(open("clean.csv", "w", encoding="utf-8"))
    num_written = 0
    w.writerow(
        ['comment_id', 'comment_created', 'body', 'score', 'action', 'action_created', 'mod', 'final_action',
         'final_action_mod', 'final_action_created', 'report', 'report_count', 'conflict_status'])
    for comment in comments:
        for action in comment['actions'].values():
            final_action = get_most_recent_action(comment)
            cid = comment['cid']
            action_name = action['action']
            action_created = action['action_created']
            mod = action['moderator']
            body = comment['body']
            ups = int(comment['ups'])
            comment_created = comment['comment_created']
            final_action_name = final_action['action']
            final_action_mod = final_action['moderator']
            final_action_created = final_action['action_created']
            if get_number_of_mods(comment) > 2:
                conflicted = 'complex'
            elif detect_conflict(comment):
                conflicted = 'conflict'
            elif detect_concordance(comment):
                conflicted = 'concordance'
            else:
                conflicted = 'none'
            for report in comment['reports'].values():
                report_name = report['report']
                count = report['count']
                w.writerow([cid, comment_created, body, ups, action_name, action_created, mod, final_action_name,
                            final_action_mod, final_action_created, report_name, count, conflicted])
                num_written += 1
                '''
                if num_written > 50:
                    return
                '''


def get_report_number_vector(comments):
    result = []
    for comment in comments:
        total_reports = 0
        for report in comment['reports'].values():
            total_reports += report['count']
        result.append(total_reports)
    return np.array(result)


def pandas_report_averages(index_str, comment_type_array):
    counts_array = []
    for ct in comment_type_array:
        cc = get_report_number_vector(ct)
        counts_array.append(cc)
    df = pd.DataFrame(counts_array, index_str)
    df1 = df.T
    df1.columns = index_str
    a = df1.describe()
    means = a.loc['mean'].values.tolist()
    stdevs = a.loc['std'].values.tolist()
    counts = a.loc['count'].values.tolist()
    sem = df.sem(axis=1).values.tolist()
    index = np.arange(len(df1.columns))

    conf_intervals = []
    for i in range(len(means)):
        ci = 1.96 * stdevs[i] / (counts[i] ** (0.5))
        conf_intervals.append(ci)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xticks(index)
    ax.set_xticklabels(df1.columns)
    ax.set_title('Average number of reports per comment')
    plt.bar(index, means, yerr=sem, capsize=10)
    plt.tight_layout()
    plt.show()


def race_vs_revisit(conflict_comments):
    race = []
    revisit = []
    for comment in conflict_comments:
        first_action = list(comment['actions'].values())[0]
        first_action = parser.parse(first_action['action_created'])
        second_action = list(comment['actions'].values())[1]
        second_action = parser.parse(second_action['action_created'])
        delta = abs(first_action - second_action)
        if delta > dt.timedelta(minutes=30):
            revisit.append(comment)
        else:
            race.append(comment)
    return race, revisit


def plot_proportions(index_str, proportions_array):
    df = pd.DataFrame(proportions_array,
                      index=index_str)
    df.plot(kind='bar', title='Report type proportions')
    plt.show()


def analyze_conflict_and_concordance(comments):
    complex_comments, concordance_comments, conflict_comments, neither_comments = categorize_comments_by_mod_action(
        comments)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Action category statistics')
    ax.set_xlabel('action category')
    ax.set_ylabel('comment count')
    count_x = ['complex', 'conflict', 'concordance', 'normal']
    count_y = [len(complex_comments), len(conflict_comments), len(concordance_comments), len(neither_comments)]
    print(count_x)
    print(count_y)
    ax.bar(count_x, count_y)
    for i, v in enumerate(count_y):
        ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
    plt.show()

    complex_avg = get_average_number_of_reports(complex_comments)
    conflict_avg = get_average_number_of_reports(conflict_comments)
    concordance_avg = get_average_number_of_reports(concordance_comments)
    neither_avg = get_average_number_of_reports(neither_comments)

    pandas_report_averages(['complex', 'conflict', 'concordance', 'normal'],
                           [complex_comments, conflict_comments, concordance_comments, neither_comments])

    complex_max = get_max_reports(complex_comments)
    conflict_max = get_max_reports(conflict_comments)
    concordance_max = get_max_reports(concordance_comments)
    neither_max = get_max_reports(neither_comments)

    complex_proportions = get_report_proportions(complex_comments)
    conflict_proportions = get_report_proportions(conflict_comments)
    concordance_proportions = get_report_proportions(concordance_comments)
    neither_proportions = get_report_proportions(neither_comments)
    plot_proportions(['complex', 'conflict', 'concordance', 'normal'],
                     [complex_proportions, conflict_proportions, concordance_proportions, neither_proportions])
    race, revisit = race_vs_revisit(conflict_comments)
    pandas_report_averages(['race', 'revisit'], [race, revisit])
    race_proportions = get_report_proportions(race)
    revisit_proportions = get_report_proportions(revisit)
    plot_proportions(['race', 'revisit'], [race_proportions, revisit_proportions])


def categorize_comments_by_mod_action(comments):
    conflict_comments = []
    concordance_comments = []
    complex_comments = []
    neither_comments = []
    for comment in comments:
        if get_number_of_mods(comment) > 2:
            complex_comments.append(comment)
        elif detect_conflict(comment):
            conflict_comments.append(comment)
        elif detect_concordance(comment):
            concordance_comments.append(comment)
        else:
            neither_comments.append(comment)
    return complex_comments, concordance_comments, conflict_comments, neither_comments


def sample_comments(comments, number):
    random.shuffle(comments)
    result = []
    for i in range(number):
        result.append(comments[i])
    return result


def get_report_proportions(comments):
    report_counts = {}
    for comment in comments:
        for report in comment['reports'].values():
            report_name = report['report']
            if report_name not in report_counts:
                report_counts[report_name] = 0.0
            report_counts[report_name] += 1.0
    result = {}
    for report, count in report_counts.items():
        proportion = count / len(comments)
        if proportion > 0.01:
            result[report] = proportion
    return result


def get_average_number_of_reports(comments):
    total_reports = 0.0
    for comment in comments:
        for report in comment['reports'].values():
            total_reports += report['count']
    return total_reports / len(comments)


def get_max_reports(comments):
    max_reports = -1
    worst_comment = None
    for comment in comments:
        total_reports = 0.0
        for report in comment['reports'].values():
            total_reports += report['count']
        if total_reports > max_reports:
            max_reports = total_reports
            worst_comment = comment
    return max_reports


def get_concordance_vs_conflict_samples(comments, sample_size=100):
    complex_comments, concordance_comments, conflict_comments, neither_comments = categorize_comments_by_mod_action(
        comments)
    concordance_sample = sample_comments(concordance_comments, sample_size)
    conflict_sample = sample_comments(conflict_comments, sample_size)
    w = csv.writer(open("concordance.csv", "w", encoding="utf-8"))
    w.writerow(['comment_id, comment_body', 'last_action'])
    for c in concordance_sample:
        action = get_most_recent_action(c)
        w.writerow([c['cid'], c['body'], action['action']])
    w = csv.writer(open("conflict.csv", "w", encoding="utf-8"))
    w.writerow(['comment_id, comment_body', 'last_action'])
    for c in conflict_sample:
        action = get_most_recent_action(c)
        w.writerow([c['cid'], c['body'], action['action']])


def get_bag_of_words_for_comments(comments, proportion=True):
    word_freq_counts = {}
    for c in comments:
        body = c['body']
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(body.lower())
        filtered_words = set([w for w in word_tokens if w not in stop_words])
        for fw in filtered_words:
            if fw not in word_freq_counts:
                word_freq_counts[fw] = 0.0
            word_freq_counts[fw] += 1.0
    result = {}
    omit_chars = ['/', '*']
    omit_words = [')', ',', '?', '.', "'s"]
    for w, count in word_freq_counts.items():
        skip = False
        for ow in omit_words:
            if w == ow:
                skip = True
                break
        for oc in omit_chars:
            if w.find(oc) != -1:
                skip = True
                break
        if skip:
            continue
        if proportion:
            result[w] = count / len(comments)
        else:
            result[w] = count
    return result


def get_approve_remove_comments(comments):
    approved = []
    removed = []
    for c in comments:
        action = get_most_recent_action(c)['action']
        if action in APPROVE_ACTIONS:
            approved.append(c)
        else:
            removed.append(c)
    return approved, removed


def get_odds_ratios(comments1, comments2, frequency_threshold, or_threshold):
    all_words = set(comments1.keys()).union(set(comments2.keys()))
    result = {}
    for w in all_words:
        if w not in comments1:
            c1_freq = 0.0
        else:
            c1_freq = comments1[w]
        if w not in comments2:
            c2_freq = 0.0
        else:
            c2_freq = comments2[w]
        if c1_freq > frequency_threshold or c2_freq > frequency_threshold:
            if c1_freq == 0.0:
                if c2_freq == 0.0:
                    odds_ratio = 0.0
                else:
                    continue
                    odds_ratio = -float('inf')
            else:
                if c2_freq == 0.0:
                    continue
                    odds_ratio = float('inf')
                else:
                    odds_ratio = log(c1_freq/ c2_freq, 2)
            if abs(odds_ratio) > or_threshold:
                result[w] = odds_ratio
    return result


def plot_odds_ratio(odds_r, labels, top_k=25):
    sorted_odds = dict(sorted(odds_r.items(), key=lambda item: item[1]))
    lo_x = list(reversed(list(sorted_odds.keys())[:top_k]))
    lo_y = [abs(sorted_odds[k]) for k in lo_x]
    hi_x = list(sorted_odds.keys())[-top_k:]
    hi_y = [abs(sorted_odds[k]) for k in hi_x]
    fig, (ax_lo, ax_hi) = plt.subplots(ncols=2)
    ax_lo.barh(lo_x, lo_y)
    ax_lo.set_title('more likely to have been {}'.format(labels[1]))
    ax_hi.barh(hi_x, hi_y)
    ax_hi.set_title('more likely to have been {}'.format(labels[0]))
    fig.suptitle('Odds ratio comparisons')


def get_actions_for_moderator(actions, who):
    result = []
    for action in actions:
        if action['moderator'] == who:
            result.append(action)
    return result


def get_self_conflicted(comments, who):
    normal = []
    conflicted = []
    for comment in comments:
        actions = get_actions_for_moderator(comment['actions'].values(), who)
        if len(actions) == 0:
            continue
        elif len(actions) == 1:
            normal.append(comment)
        else:
            approved = False
            removed = False
            for action in actions:
                if action['action'] in APPROVE_ACTIONS:
                    approved = True
                if action['action'] in REMOVE_ACTIONS:
                    removed = True
            if approved and removed:
                conflicted.append(comment)
            else:
                normal.append(comment)
    return normal, conflicted


def bag_of_words_compare(comments, word_freq_cutoff=5, frequency_threshold=0.005, or_threshold=0.75):
    complex_comments, concordance_comments, conflict_comments, neither_comments = categorize_comments_by_mod_action(
        comments)

    approved_normal, removed_normal = get_approve_remove_comments(neither_comments)

    approved_sample = sample_comments(approved_normal, 10000)
    removed_sample = sample_comments(removed_normal, 10000)

    approved_freq = get_bag_of_words_for_comments(approved_sample)
    removed_freq = get_bag_of_words_for_comments(removed_sample)
    ar_or = get_odds_ratios(approved_freq, removed_freq, frequency_threshold, or_threshold)
    plot_odds_ratio(ar_or, ['accepted by moderator', 'rejected by moderator'])
    approved_count = get_bag_of_words_for_comments(approved_sample, proportion=False)
    removed_count = get_bag_of_words_for_comments(removed_sample, proportion=False)
    sage_pair(approved_count, removed_count, 'Approve', 'Remove')
    plt.show()

    # conflict_freq = get_bag_of_words_for_comments(conflict_comments)
    # concordance_freq = get_bag_of_words_for_comments(concordance_comments)
    
    race_conc, revisit_conc = race_vs_revisit(concordance_comments)
    race_conf, revisit_conf = race_vs_revisit(conflict_comments)
    race_conc_freq = get_bag_of_words_for_comments(race_conc)
    race_conf_freq = get_bag_of_words_for_comments(race_conf)
    cc_race_or = get_odds_ratios(race_conf_freq, race_conc_freq, frequency_threshold, or_threshold)
    plot_odds_ratio(cc_race_or, ['race condition and conflicted', 'race condition and concordant'])
    race_conc_count = get_bag_of_words_for_comments(race_conc, proportion=False)
    race_conf_count = get_bag_of_words_for_comments(race_conf, proportion=False)
    sage_pair(race_conc_count, race_conf_count, 'Race concordance', 'Race conflicted')
    plt.show()
    
    revisit_conc_freq = get_bag_of_words_for_comments(revisit_conc)
    revisit_conf_freq = get_bag_of_words_for_comments(revisit_conf)
    cc_revisit_or = get_odds_ratios(revisit_conf_freq, revisit_conc_freq, frequency_threshold, or_threshold)
    plot_odds_ratio(cc_revisit_or, ['revisit and conflicted', 'revisit and concordant'])
    revisit_conc_count = get_bag_of_words_for_comments(revisit_conc, proportion=False)
    revisit_conf_count = get_bag_of_words_for_comments(revisit_conf, proportion=False)
    sage_pair(revisit_conc_count, revisit_conf_count, 'Revisit concordance', 'Revisit conflicted')
    plt.show()

    self_normal, self_conflicted = get_self_conflicted(neither_comments, 'barber5')
    self_normal_count = get_bag_of_words_for_comments(self_normal, proportion=False)
    self_conflicted_count = get_bag_of_words_for_comments(self_conflicted, proportion=False)
    sage_pair(self_normal_count, self_conflicted_count, 'Normal for barber5', 'Self conflicted for barber5')
    plt.show()


def sage_pair(base_counts, child_counts, base_title, child_title):
    vocab = [word for word, count in Counter(child_counts).most_common(1000)]
    x_child = np.array([child_counts[word] for word in vocab])
    for word in vocab:
        if word not in base_counts:
            base_counts[word] = 0
    x_base = np.array([base_counts[word] for word in vocab]) + 1.
    mu = np.log(x_base) - np.log(x_base.sum())
    eta = sage.estimate(x_child, mu)
    print(sage.top_k(eta, vocab, 20))
    top_k = sage.top_k(eta, vocab, 20)
    print(sage.bottom_k(eta, vocab, 20))
    bottom_k = sage.bottom_k(eta, vocab, 20)
    (fig, ax) = plt.subplots()
    ax.set_title('{} vs {} histogram of SAGE scores'.format(base_title, child_title))
    ax.set_xlabel('SAGE score')
    ax.set_ylabel('count of terms with score')
    ax.hist(eta, 20)

    (fig, ax) = plt.subplots()
    ax.set_title('{} vs {} SAGE score distribution'.format(base_title, child_title))
    ax.set_xlabel('terms')
    ax.set_ylabel('SAGE score')
    ax.plot(sorted(eta))

    lo_x = list(reversed(list(bottom_k.keys())))
    lo_freqs = []
    for w in bottom_k:
        if w not in base_counts:
            bc = 0
        else:
            bc = base_counts[w]
        if w not in child_counts:
            cc = 0
        else:
            cc = child_counts[w]
        lo_freqs.append(bc + cc)
    lo_y = list(reversed(list(bottom_k.values())))
    hi_x = list(reversed(list(top_k.keys())))
    hi_freqs = []
    for w in top_k:
        if w not in base_counts:
            bc = 0
        else:
            bc = base_counts[w]
        if w not in child_counts:
            cc = 0
        else:
            cc = child_counts[w]
        hi_freqs.append( bc + cc)
    hi_y = list(reversed(list(top_k.values())))
    fig, axs = plt.subplots(ncols=2, nrows=2)
    axs[0][0].barh(lo_x, lo_y)
    axs[0][0].set_title('more likely to have been {}'.format(base_title))
    axs[0][1].barh(hi_x, hi_y)
    axs[0][1].set_title('more likely to have been {}'.format(child_title))
    axs[1][0].barh(lo_x, lo_freqs)
    axs[1][0].set_title('word frequencies')
    axs[1][1].barh(hi_x, hi_freqs)
    axs[1][1].set_title('word frequencies')
    fig.suptitle('SAGE score comparisons')


def moderator_analysis(comments, min_action_threshold=10):
    complex_comments, concordance_comments, conflict_comments, neither_comments = categorize_comments_by_mod_action(
        comments)
    race_conf, revisit_conf = race_vs_revisit(conflict_comments)
    race_conc, revisit_conc = race_vs_revisit(concordance_comments)
    # first get a baseline for each mod
    mod_baseline_idx = {}
    for comment in neither_comments:
        action = get_most_recent_action(comment)
        mod = action['moderator']
        action_name = action['action']
        if mod not in mod_baseline_idx:
            mod_baseline_idx[mod] = []
        if action_name in APPROVE_ACTIONS:
            mod_baseline_idx[mod].append(1)
        else:
            mod_baseline_idx[mod].append(0)


    # then see if there's a sig diff when they are in conflict
    # i.e. they are acting unusually on comments where they are in conflict
    mod_conflict_idx = {}
    for comment in race_conf:
        mod_action_dict = get_most_recent_action_by_mod(comment)
        for mod, action in mod_action_dict.items():
            if mod not in mod_conflict_idx:
                mod_conflict_idx[mod] = []
            action_name = action['action']
            if action_name in APPROVE_ACTIONS:
                mod_conflict_idx[mod].append(1)
            else:
                mod_conflict_idx[mod].append(0)
    bonferonni_m = 0.0
    for mod in set(mod_conflict_idx.keys()).intersection(set(mod_baseline_idx.keys())):
        if len(mod_conflict_idx[mod]) < min_action_threshold or len(mod_baseline_idx[mod]) < min_action_threshold:
            continue
        bonferonni_m += 1.0
    corrected_alpha = 0.05 / bonferonni_m
    print('bonferonni alpha for alpha=0.05 is {}'.format(corrected_alpha))
    mods = []
    bap = []
    brp = []
    bta = []
    cap = []
    crp = []
    cta = []
    tvals = []
    for mod in set(mod_conflict_idx.keys()).intersection(set(mod_baseline_idx.keys())):
        baseline_actions = np.array(mod_baseline_idx[mod])
        conf_actions = np.array(mod_conflict_idx[mod])
        if len(conf_actions) < min_action_threshold or len(mod_baseline_idx[mod]) < min_action_threshold:
            continue
        (stat, p_value) = stats.ttest_ind(a=baseline_actions, b=conf_actions, equal_var=False)
        if p_value < corrected_alpha:
            sig_chars = '***'
        else:
            sig_chars = ''
        mods.append(mod)
        bap.append(round(baseline_actions.mean(), 2))
        brp.append(round(1-baseline_actions.mean(), 2))
        bta.append(len(baseline_actions))
        cap.append(round(conf_actions.mean(), 2))
        crp.append(round(1-conf_actions.mean(), 2))
        cta.append(len(conf_actions))
        tvals.append('t({})={}, p={}{}'.format(len(baseline_actions) + len(conf_actions) - 2, round(stat, 2), p_value, sig_chars))
        print('\n\n\n\n----------------------------------------')
        print('stats for {}'.format(mod))
        print('approval proportion baseline {}'.format(baseline_actions.mean()))
        print('removal proportion baseline {}'.format(1-baseline_actions.mean()))
        print('baseline total actions {}'.format(len(baseline_actions)))
        print('approval proportion conflict-race {}'.format(conf_actions.mean()))
        print('remove proportion conflict-race {}'.format(1-conf_actions.mean()))
        print('conflict-race total actions {}'.format(len(conf_actions)))
        print('t={}, p-value={}{}'.format(stat, p_value, sig_chars))
        print('----------------------------------------')
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Moderator', 'Baseline approval proportion', 'Baseline removal proportion', 'Baseline total actions', 'Conflict approval proportion', 'Conflict removal proportion', 'Conflict total actions', 't-test'],
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=[mods,  # 1st column
                           bap,
                           brp,
                           bta,
                           cap,
                           crp,
                           cta,
                           tvals
                           ],
                   line_color='darkslategray',
                   fill_color='lightcyan',
                   align='left'))
    ])

    fig.show()


def split_data(negatives, positives):
    X = []
    y = []
    for c in negatives:
        X.append(c)
        y.append(0)
    for c in positives:
        X.append(c)
        y.append(1)
    return train_test_split(X, y, test_size=0.10)


def get_document_vectorizer(X_train, cutoff=10):
    word_freqs = get_bag_of_words_for_comments(X_train, proportion=False)
    documents = []
    vectorizer = CountVectorizer()
    for c in X_train:
        current = []
        body = c['body']
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(body.lower())
        filtered_words = set([w for w in word_tokens if w not in stop_words])
        for fw in filtered_words:
            if fw not in word_freqs or word_freqs[fw] < cutoff:
                continue
            current.append(fw)
        # current = np.array(current)
        documents.append(' '.join(current))
    vectorizer.fit(documents)
    return vectorizer


def word_vectorize_comments(comments, vectorizer):
    # word_freqs = get_bag_of_words_for_comments(comments, proportion=False)
    documents = []
    for c in comments:
        current = []
        body = c['body']
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(body.lower())
        filtered_words = set([w for w in word_tokens if w not in stop_words])
        for fw in filtered_words:
            current.append(fw)
        # current = np.array(current)
        documents.append(' '.join(current))
    return vectorizer.transform(documents).toarray()


def get_report_features(comments):
    X = []
    for c in comments:
        politics = 0
        info_quality = 0
        incivility = 0
        other = 0
        bot = 0
        spam = 0
        none = 0
        for r in c['reports'].values():
            report = r['report']
            count = r['count']
            if report == 'politics':
                politics += count
            elif report == 'incivility':
                incivility += count
            elif report == 'information quality':
                info_quality += count
            elif report == 'spam':
                spam += count
            elif report == 'other':
                other += count
            elif report == 'bot generated report':
                bot += count
            elif report == 'no report':
                none += count
        X.append([politics, info_quality, incivility, other, bot, spam, none])
    return X


def get_ups_feature(comments):
    X = []
    for c in comments:
        ups = c['ups']
        X.append([ups])
    return X


def index_comments_by_author(comments):
    author_idx = {}
    for c in comments:
        aid = c['aid']
        if aid not in author_idx:
            author_idx[aid] = []
        author_idx[aid].append(c)
    return author_idx


def count_comments_before(comment, comment_idx):
    aid = comment['aid']
    when = comment['comment_created']
    if aid not in comment_idx:
        return 0
    count = 0
    for c in comment_idx[aid]:
        c_when = c['comment_created']
        if c_when < when:
            count += 1
    return count


def get_prior_actions_features(comments, neither_comments, concordance_comments, conflict_comments, complex_comments):
    approved_normal, removed_normal = get_approve_remove_comments(neither_comments)
    a_idx = index_comments_by_author(approved_normal)
    r_idx = index_comments_by_author(removed_normal)
    conc_idx = index_comments_by_author(concordance_comments)
    conf_idx = index_comments_by_author(conflict_comments)
    comp_idx = index_comments_by_author(complex_comments)
    X = []
    for c in comments:
        a_prior = count_comments_before(c, a_idx)
        r_prior = count_comments_before(c, r_idx)
        conc_prior = count_comments_before(c, conc_idx)
        conf_prior = count_comments_before(c, conf_idx)
        comp_prior = count_comments_before(c, comp_idx)
        X.append([a_prior, r_prior, conc_prior, conf_prior, comp_prior])
    return X


def featurize(vectorizer, X, neither_comments, concordance_comments, conflict_comments, complex_comments):
    X_train_vectorized = word_vectorize_comments(X, vectorizer)
    X_reports = get_report_features(X)
    X_ups = get_ups_feature(X)
    X_prior_actions = get_prior_actions_features(X, neither_comments, concordance_comments, conflict_comments,
                                                 complex_comments)
    X = []
    for i in range(len(X_train_vectorized)):
        next_x = []
        for k in X_ups[i]:
            next_x.append(k)
        for k in X_reports[i]:
            next_x.append(k)
        for k in X_prior_actions[i]:
            next_x.append(k)
        for k in X_train_vectorized[i]:
            next_x.append(k)
        X.append(np.array(next_x))
    return np.array(X)


def preprocess_data_for_conflict_concordance_classification(comments):
    complex_comments, concordance_comments, conflict_comments, neither_comments = categorize_comments_by_mod_action(
        comments)
    race_conc, revisit_conc = race_vs_revisit(concordance_comments)
    race_conf, revisit_conf = race_vs_revisit(conflict_comments)

    X_train, X_test, y_train, y_test = split_data(race_conc, race_conf)
    vectorizer = get_document_vectorizer(X_train)
    X_train = featurize(vectorizer, X_train, neither_comments, concordance_comments, conflict_comments,
                        complex_comments)
    X_test = featurize(vectorizer, X_test, neither_comments, concordance_comments, conflict_comments, complex_comments)
    return X_train, X_test, y_train, y_test, vectorizer


def classify_conflicts(comments):
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data_for_conflict_concordance_classification(comments)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    cr = classification_report(y_test, y_pred)
    print('classification report: {}'.format(cr))
    cm = confusion_matrix(y_test, y_pred)
    print('confusion matrix: {}'.format(cm))
    acc = accuracy_score(y_test, y_pred)
    print('test accuracy: {}'.format(acc))
    train_acc = accuracy_score(y_train, y_train_pred)
    print('train accuracy: {}'.format(train_acc))


def get_mod_modeling_data(comments, vectorizer):
    complex_comments, concordance_comments, conflict_comments, neither_comments = categorize_comments_by_mod_action(
        comments)
    approve_comments, remove_comments = get_approve_remove_comments(neither_comments)
    X_train, X_test, y_train, y_test = split_data(approve_comments, remove_comments)
    X_train = featurize(vectorizer, X_train, neither_comments, concordance_comments, conflict_comments,
                        complex_comments)
    X_test = featurize(vectorizer, X_test, neither_comments, concordance_comments, conflict_comments, complex_comments)
    return X_train, X_test, y_train, y_test


def get_mod_involved_comments(comments, moderator_name):
    result = []
    for comment in comments:
        for action in comment['actions'].values():
            if action['moderator'] == moderator_name:
                result.append(comment)
    return result


def get_model_for_mod(comments, moderator_name, vectorizer):
    mod_comments = get_mod_involved_comments(comments, moderator_name)
    X_train, X_test, y_train, y_test = get_mod_modeling_data(mod_comments, vectorizer)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    cr = classification_report(y_test, y_pred)
    print('classification report\n{}'.format(cr))
    cm = confusion_matrix(y_test, y_pred)
    print('confusion matrix\n{}'.format(cm))
    acc = accuracy_score(y_test, y_pred)
    print('test accuracy: {}'.format(acc))
    train_acc = accuracy_score(y_train, y_train_pred)
    print('train accuracy: {}'.format(train_acc))
    return model


def get_active_mods(comments, threshold):
    activity_idx = {}
    for comment in comments:
        for action in comment['actions'].values():
            modname = action['moderator']
            if modname not in activity_idx:
                activity_idx[modname] = 0
            activity_idx[modname] += 1
    result = []
    for mod, actions in activity_idx.items():
        if actions > threshold:
            result.append(mod)
    return result


def classify_conflicts_with_mod_models(comments, mod_active_threshold=10000):
    active_mods = get_active_mods(comments, mod_active_threshold)
    X_train, X_test, y_train, y_test, vectorizer = preprocess_data_for_conflict_concordance_classification(comments)
    print(active_mods)
    models = {}
    for mod in active_mods:
        model = get_model_for_mod(comments, mod, vectorizer)
        print('trained model for {}'.format(mod))
        models[mod] = model

    y_preds = {}
    print('----------------------------------------------------')
    for mod, model in models.items():
        y_preds[mod] = model.predict(X_test)
    for vote_threshold in [0.0, 0.25, 0.33, 0.5, 0.75, 0.85, 0.95, 1.0]:
        print('with vote threshold: {}'.format(vote_threshold))
        y_pred_aggregate = []
        for i in range(len(y_test)):
            # print('\n\ndata item {}'.format(i))
            num0 = 0
            num1 = 0
            for mod, y_p in y_preds.items():
                prediction = y_p[i]
                if prediction == 0:
                    num0 += 1
                else:
                    num1 += 1
                # print('mod {}, prediction (0=approve, 1=remove): {}'.format(mod, prediction))
            if max(num0, num1) / float(num0 + num1) >= vote_threshold:
                y_pred_aggregate.append(0)
            else:
                y_pred_aggregate.append(1)
            # print('--true label (0=concordance, 1=conflict): {}, number of approve votes: {}, number of remove votes: {}'.format(y_test[i], num0, num1))
        cr = classification_report(y_test, y_pred_aggregate)
        print('classification report\n{}'.format(cr))
        cm = confusion_matrix(y_test, y_pred_aggregate)
        print('confusion matrix\n{}'.format(cm))
        acc = accuracy_score(y_test, y_pred_aggregate)
        print('test accuracy: {}'.format(acc))


if __name__ == "__main__":
    comments_all = get_all_comments_from_db()
    comments_all = code_reports(comments_all)
    moderator_analysis(comments_all)
    print('got {} comments'.format(len(comments_all)))
    count_removal_reasons(comments_all)
    mod_action_statistics(comments_all)
    # bag_of_words_compare(comments_all)
    # classify_conflicts(comments_all)
    classify_conflicts_with_mod_models(comments_all)

    analyze_conflict_and_concordance(comments_all)
    mod_conflict_statistics(comments_all)
    compare_conflict_to_non_conflict(comments_all)
    # fix_none_reports(comments_all)
    # analyze_report_long_tail(comments_all)
    # last_entry(comments_all)
    # get_last_action_for_each_mod(comments_all)
