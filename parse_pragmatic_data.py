import re

def extract_limits():
    with open('docs/customer_range_report.txt', 'r') as f:
        content = f.read()

    print(content[-2000:])

if __name__ == '__main__':
    extract_limits()
