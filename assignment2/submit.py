import time
import os
import sys
import argparse
import json
import datetime
from bonnie.submission import Submission

# open stdout unbuffered to automatically flush prints
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

REFRESH_TIME = 3.0

LATE_POLICY = \
"""Late Policy:
  \"I have read the late policy for CS6475. I understand that only my last
  commit before the late submission deadline will be accepted and that late
  penalties apply if any part of the assignment is submitted late.\"
"""

HONOR_PLEDGE = "Honor Pledge:\n\n  \"I have neither given nor received aid on this assignment.\"\n"


def require_pledges():
    print(LATE_POLICY)
    ans = raw_input("Please type 'yes' to agree and continue>")
    if ans != "yes":
        raise RuntimeError("You must accept the late policy to submit your assignment.")

    print
    print(HONOR_PLEDGE)
    ans = raw_input("Please type 'yes' to agree and continue>")
    if ans != "yes":
        raise RuntimeError("You must accept the honor pledge to submit your assignment.")
    print


def main(args):

    require_pledges()

    print "Submitting files..."
    submission = Submission('cs6475', args.quiz,
                            filenames=args.files,
                            environment=args.environment,
                            provider=args.provider)

    print "\nWaiting for results. (Refresh every {0} seconds)".format(REFRESH_TIME)
    time.sleep(REFRESH_TIME)
    while not submission.poll():
        print "    Re-trying in {0} seconds...".format(REFRESH_TIME)
        time.sleep(REFRESH_TIME)
    print "    Done!"

    print "\nResults:\n--------"
    if submission.feedback():

        if submission.console():
                print submission.console()

        timestamp = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now())
        filename = "%s-result-%s.json" % (args.quiz, timestamp)

        with open(filename, "w") as fd:
            json.dump(submission.feedback(), fd, indent=4, separators=(',', ': '))

        print("\n(Details available in %s)\n" % filename)

    elif submission.error_report():
        error_report = submission.error_report()
        print(json.dumps(error_report, indent=4))

    else:
        print("Unknown error.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submits code to the Udacity site.')
    parser.add_argument('part', choices=['assignment2', 'writeup'])
    parser.add_argument('--provider', choices=['gt', 'udacity'], default='gt')
    parser.add_argument('--environment', choices=['local', 'development', 'staging', 'production'], default='production')

    args = parser.parse_args()

    if args.part == 'assignment2':
        setattr(args, 'quiz', 'assignment2')
        setattr(args, 'files', ["assignment2.py"])
    else:
        setattr(args, 'quiz', 'writeup2')
        setattr(args, 'files', ["assignment2.pdf"])

    main(args)