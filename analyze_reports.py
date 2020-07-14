import pickle
import random
import sqlite3
import datetime as dt
from math import log

from dateutil import parser
import scipy.stats as stats
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


def get_bag_of_words_for_comments(comments):
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
    for w, count in word_freq_counts.items():
        result[w] = count / len(comments)
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
    plt.show()
    print(sorted_odds)


def bag_of_words_compare(comments, word_freq_cutoff=5, frequency_threshold=0.005, or_threshold=0.75):
    complex_comments, concordance_comments, conflict_comments, neither_comments = categorize_comments_by_mod_action(
        comments)
    approved_normal, removed_normal = get_approve_remove_comments(neither_comments)
    race_conc, revisit_conc = race_vs_revisit(concordance_comments)
    race_conf, revisit_conf = race_vs_revisit(conflict_comments)

    approved_sample = sample_comments(approved_normal, 5000)
    removed_sample = sample_comments(removed_normal, 5000)

    approved_freq = get_bag_of_words_for_comments(approved_sample)
    removed_freq = get_bag_of_words_for_comments(removed_sample)
    conflict_freq = get_bag_of_words_for_comments(conflict_comments)
    concordance_freq = get_bag_of_words_for_comments(concordance_comments)
    race_conc_freq = get_bag_of_words_for_comments(race_conc)
    race_conf_freq = get_bag_of_words_for_comments(race_conf)
    revisit_conc_freq = get_bag_of_words_for_comments(revisit_conc)
    revisit_conf_freq = get_bag_of_words_for_comments(revisit_conf)

    ar_or = get_odds_ratios(approved_freq, removed_freq, frequency_threshold, or_threshold)
    cc_race_or = get_odds_ratios(race_conf_freq, race_conc_freq, frequency_threshold, or_threshold)
    cc_revisit_or = get_odds_ratios(revisit_conf_freq, revisit_conc_freq, frequency_threshold, or_threshold)
    plot_odds_ratio(ar_or, ['accepted by moderator', 'rejected by moderator'])
    plot_odds_ratio(cc_race_or, ['race condition and conflicted', 'race condition and concordant'])
    plot_odds_ratio(cc_revisit_or, ['revisit and conflicted', 'revisit and concordant'])


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



if __name__ == "__main__":
    comments_all = get_all_comments_from_db()
    comments_all = code_reports(comments_all)
    moderator_analysis(comments_all)
    # print('got {} comments'.format(len(comments_all)))
    # count_removal_reasons(comments_all)
    # mod_action_statistics(comments_all)
    # analyze_conflict_and_concordance(comments_all)
    # fix_none_reports(comments_all)
    # analyze_report_long_tail(comments_all)

    # mod_conflict_statistics(comments_all)
    # compare_conflict_to_non_conflict(comments_all)
    # last_entry(comments_all)
    # get_last_action_for_each_mod(comments_all)
