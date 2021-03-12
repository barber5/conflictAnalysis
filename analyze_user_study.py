import csv
import os
from math import log

NUM_PARTICIPANTS = 10


def get_file_names():
    return filter(lambda x: x[6] != '.', [f'study/{fn}' for fn in os.listdir('study/')])


def get_answers_for_comment(comment_number, row, headers):
    result = {'comment_number': comment_number}
    for i in range(7):
        index = comment_number * 7 + 2 + i
        header = headers[index]
        value = row[index]
        result[header] = value
    return result


def read_csv_file(fname):
    with open(fname) as csvfile:
        csvreader = csv.reader(csvfile)
        headers = []
        for row in csvreader:
            if len(headers) == 0:
                headers = row
        data_dict = {
            headers[0]: row[0],
            headers[1]: row[1],
            'decisions': {
                'concordance': [],
                'conflict': [],
                'approve': [],
                'remove': []
            }
        }
        for i in range(5):
            data_dict['decisions']['conflict'].append(get_answers_for_comment(i, row, headers))
        for i in range(5):
            data_dict['decisions']['concordance'].append(get_answers_for_comment(i + 5, row, headers))
        for i in range(3):
            data_dict['decisions']['approve'].append(get_answers_for_comment(i + 10, row, headers))
        for i in range(3):
            data_dict['decisions']['remove'].append(get_answers_for_comment(i + 13, row, headers))
        return data_dict


def get_data():
    data = []
    fnames = get_file_names()
    for fname in fnames:
        data.append(read_csv_file(fname))
    return data


def extract_answer_data(q, question_idx, who, the_type):
    cn = q['comment_number']
    if cn not in question_idx[the_type]:
        question_idx[the_type][cn] = {}
    for k, v in q.items():
        if k == 'comment_number':
            continue
        if k not in question_idx[the_type][cn]:
            question_idx[the_type][cn][k] = []
        question_idx[the_type][cn][k].append((who, v))


def get_action_entropy(q_data):
    approve_count = 0
    remove_count = 0
    for (who, response) in q_data['Which action would you take on the comment above']:
        if response == 'Approve':
            approve_count += 1
        elif response == 'Remove':
            remove_count += 1

    approve_prob = approve_count / (approve_count + remove_count)
    remove_prob = remove_count / (approve_count + remove_count)
    if approve_count == 0.0 or remove_prob == 0.0:
        entropy = 0
    else:
        entropy = -1 * approve_prob * log(approve_prob, 2) - remove_prob * log(remove_prob, 2)
    return entropy


def get_self_disagreement(q_data, orig):
    num_disagreements = 0
    for (who, response) in q_data['Which action would you take on the comment above']:
        if response != orig:
            num_disagreements += 1
    return num_disagreements


def get_difficulty(q_data):
    total_resp = 0
    for (who, response) in q_data['Rate the extent of your agreement or disagreement with the following statements [It is easy to decide what action to take for this piece of content]']:
        if response == 'Strongly disagree':
            resp_code = 5
        elif response == 'Disagree':
            resp_code = 5
        elif response == 'Agree':
            resp_code = 2
        elif response == 'Strongly agree':
            resp_code = 1
        else:
            resp_code = 3
        total_resp += resp_code
    return total_resp


def analyze_data(data):
    question_idx = group_by_question(data)
    total_ent = 0
    total_difficulty = 0
    total_questions = NUM_PARTICIPANTS * 5
    for q, q_data in question_idx['conflict'].items():
        entropy = get_action_entropy(q_data)
        difficulty = get_difficulty(q_data)
        total_difficulty += difficulty
        total_ent += entropy
        print(f'entropy {entropy}')
        print(f'difficulty {difficulty} {difficulty/NUM_PARTICIPANTS}')
    print(f'avg entropy for conflict is {total_ent/5}')
    print(f'avg difficulty for conflict is {total_difficulty / total_questions}')
    total_ent = 0
    total_difficulty = 0
    for q, q_data in question_idx['concordance'].items():
        entropy = get_action_entropy(q_data)
        difficulty = get_difficulty(q_data)
        total_difficulty += difficulty
        total_ent += entropy
        print(f'entropy {entropy}')
        print(f'difficulty {difficulty} {difficulty/NUM_PARTICIPANTS}')
    print(f'avg entropy for concordance is {total_ent / 5}')
    print(f'avg difficulty for concordance is {total_difficulty / total_questions}')
    total_questions = NUM_PARTICIPANTS*3
    total_disagreements = 0
    for q, q_data in question_idx['approve'].items():
        num_disagreements = get_self_disagreement(q_data, 'Approve')
        total_disagreements += num_disagreements
    print(f'approve num disagreements was {total_disagreements} of {total_questions} {total_disagreements / total_questions}')
    total_disagreements = 0
    for q, q_data in question_idx['remove'].items():
        num_disagreements = get_self_disagreement(q_data, 'Remove')
        total_disagreements += num_disagreements
    print(f'remove num disagreements was {total_disagreements} of {total_questions} {total_disagreements / total_questions}')


def group_by_question(data):
    question_idx = {
        'conflict': {},
        'concordance': {},
        'approve': {},
        'remove': {}
    }
    for dd in data:
        who = dd['What is your Reddit username']
        for q in dd['decisions']['conflict']:
            extract_answer_data(q, question_idx, who, 'conflict')
        for q in dd['decisions']['concordance']:
            extract_answer_data(q, question_idx, who, 'concordance')
        for q in dd['decisions']['approve']:
            extract_answer_data(q, question_idx, who, 'approve')
        for q in dd['decisions']['remove']:
            extract_answer_data(q, question_idx, who, 'remove')
    return question_idx


if __name__ == "__main__":
    the_data = get_data()
    analyze_data(the_data)
