import os

from analyze_reports import get_all_comments_from_db
import pickle

def get_unique_report_reasons(comments):
    unique_reports = set([])
    for comment in comments:
        for report in comment['reports'].values():
            report_name = report['report']
            unique_reports.add(report_name)
    return unique_reports


def auto_map_it(all_reports_unique):
    auto_map = {
        'c': set([]),
        'p': set([]),
        's': set([]),
        'b': set([]),
        'q': set([]),
        'n': set([]),
        'u': set([]),
        'o': set([])
    }
    incivility_words = ['civil', 'toxic', 'troll', 'racis', 'nazi', 'harass', 'stalk', 'semit', 'hate', 'misog',
                        'harass', 'xeno']
    politics_words = ['politic', 'propaganda', 'kremlin']
    spam_words = ['bot', 'spam']
    bot_words = ['501']
    quality_words = ['misinformation', 'info', 'topic', 'conspir', 'effort', 'fake']
    none_words = ['null', 'No report associated']

    for report in all_reports_unique:
        assigned = False
        for iw in incivility_words:
            if report.lower().find(iw) != -1:
                auto_map['c'].add(report)
                assigned = True
                break
        if not assigned:
            for kw in politics_words:
                if report.lower().find(kw) != -1:
                    auto_map['p'].add(report)
                    assigned = True
                    break
        if not assigned:
            for kw in spam_words:
                if report.lower().find(kw) != -1:
                    auto_map['s'].add(report)
                    assigned = True
                    break
        if not assigned:
            for kw in bot_words:
                if report.lower().find(kw) != -1:
                    auto_map['b'].add(report)
                    assigned = True
                    break
        if not assigned:
            for kw in quality_words:
                if report.lower().find(kw) != -1:
                    auto_map['q'].add(report)
                    assigned = True
                    break
        if not assigned:
            for kw in none_words:
                if report.lower().find(kw) != -1:
                    auto_map['n'].add(report)
                    assigned = True
                    break
        if not assigned:
            auto_map['u'].add(report)
    return auto_map


def print_codebook(codebook):
    for cat, reports in codebook.items():
        print('\n\n\n\n\n--------------{}---------------\n\n\n\n\n\n\n{} count'.format(cat, len(reports)))
        for report in reports:
            print(report)


def code_reports_interactively(all_reports_unique):
    if os.path.isfile('codebook.pkl'):
        with open('codebook.pkl', 'rb') as fi:
            auto_map = pickle.load(fi)
    else:
        auto_map = auto_map_it(all_reports_unique)

    i = 0
    other_length = len(auto_map['u'])
    for report in list(auto_map['u']):
        i += 1
        print('{} of {}: {}'.format(i, other_length, report))
        code = input("Code? c=incivility, q=info quality, p=politics, s=spam, n=none, b=bot, o=other")
        auto_map['u'].remove(report)
        auto_map[code].add(report)
        with open('codebook.pkl', 'wb') as fi:
            pickle.dump(auto_map, fi)


if __name__ == "__main__":
    all_comments = get_all_comments_from_db()
    all_reports = get_unique_report_reasons(all_comments)
    code_reports_interactively(all_reports)