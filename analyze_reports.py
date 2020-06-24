import pickle
import sqlite3
import datetime as dt
from dateutil import parser
from config import APPROVE_ACTIONS, REMOVE_ACTIONS
from util import authorize
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def conflict_race_vs_revisit(conflict_comments):
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
    race, revisit = conflict_race_vs_revisit(conflict_comments)
    pandas_report_averages(['race', 'revisit'], [race, revisit])
    race_proportions = get_report_proportions(race)
    revisit_proportions = get_report_proportions(revisit)
    plot_proportions(['race', 'revisit'], [race_proportions, revisit_proportions])


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


if __name__ == "__main__":
    comments_all = get_all_comments_from_db()
    comments_all = code_reports(comments_all)
    print('got {} comments'.format(len(comments_all)))
    count_removal_reasons(comments_all)
    mod_action_statistics(comments_all)
    analyze_conflict_and_concordance(comments_all)
    # fix_none_reports(comments_all)
    # analyze_report_long_tail(comments_all)

    # mod_conflict_statistics(comments_all)
    # compare_conflict_to_non_conflict(comments_all)
    # last_entry(comments_all)
    # get_last_action_for_each_mod(comments_all)
