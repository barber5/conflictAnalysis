import sqlite3
from config import APPROVE_ACTIONS, REMOVE_ACTIONS
from util import authorize
import csv

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
        if len(comment['reports']) == 0:
            if 'none' not in count_idx:
                count_idx['none'] = 0
            count_idx['none'] += 1
    for reason, count in count_idx.items():
        print('{}\t{}'.format(reason, count))


def get_most_recent_action(comment):
    # get most recent action
    most_recent_action = comment['actions'].values()[0]
    most_recent_time = comment['actions'].values()[0]['action_created']
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
    print('report reason plus final action statistics')
    for reason, count_meta in reason_action_count_idx.items():
        total_actions = 0
        for action, count in count_meta.items():
            total_actions += count
        if total_actions > COMMON_ACTION_THRESHOLD:
            for action, count in count_meta.items():
                print('{}\t{}\t{}'.format(reason, action, count))


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
    complex_avg = get_average_number_of_reports(complex_comments)
    conflict_avg = get_average_number_of_reports(conflict_comments)
    concordance_avg = get_average_number_of_reports(concordance_comments)
    neither_avg = get_average_number_of_reports(neither_comments)

    complex_max = get_max_reports(complex_comments)
    conflict_max = get_max_reports(conflict_comments)
    concordance_max = get_max_reports(concordance_comments)
    neither_max = get_max_reports(neither_comments)

    complex_proportions = get_report_proportions(complex_comments)
    conflict_proportions = get_report_proportions(conflict_comments)
    concordance_proportions = get_report_proportions(concordance_comments)
    neither_proportions = get_report_proportions(neither_comments)
    print('hi')


def get_report_proportions(comments):
    report_counts = {}
    for comment in comments:
        for report in comment['reports'].values():
            report_name = report['report']
            if report_name not in report_counts:
                report_counts[report_name] = 0.0
            report_counts[report_name] += 1.0
    result = {}
    other_total = 0.0
    for report, count in report_counts.items():
        proportion = count / len(comments)
        if proportion > 0.01:
            result[report] = proportion
        else:
            other_total += proportion
    result['Other'] = other_total
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
    analyze_conflict_and_concordance(comments_all)
    # fix_none_reports(comments_all)
    # analyze_report_long_tail(comments_all)
    # count_removal_reasons(comments_all)
    # mod_action_statistics(comments_all)
    # mod_conflict_statistics(comments_all)
    # compare_conflict_to_non_conflict(comments_all)
    # last_entry(comments_all)
    # get_last_action_for_each_mod(comments_all)
