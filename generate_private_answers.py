import json, sys
from template import calculate_reimbursement

def main():
    with open('private_cases.json') as f:
        cases = json.load(f)
    answers = []
    for entry in cases:
        ans = calculate_reimbursement(
            entry['trip_duration_days'],
            entry['miles_traveled'],
            entry['total_receipts_amount'],
        )
        answers.append(ans)
    with open('private_answers.txt', 'w') as f:
        for a in answers:
            f.write(f"{a}\n")
    print('Generated private_answers.txt with', len(answers), 'lines')

if __name__ == '__main__':
    main() 