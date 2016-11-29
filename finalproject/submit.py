import time
import os
import sys
import argparse
import json
import datetime
from bonnie.submission import Submission

# open stdout unbuffered to automatically flush prints
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

FILENAMES = {'finalproject': ["finalproject.pdf"]}
SIZE_LIMIT = 6       # maximum size of each file uploaded
REFRESH_TIME = 3.0   # number of seconds to wait during results polling

LATE_POLICY = \
"""Late Policy:\n
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


def check_files(filenames):

    if not all(map(lambda x: os.path.isfile(x), filenames)):
        raise RuntimeError("Submission Failed: One or more required files " +
                           "is missing from the current directory. Make " +
                           "sure all these files exist: {!s}".format(filenames))
    elif any(map(lambda x: os.stat(x).st_size >> 20 > SIZE_LIMIT, filenames)):
        large_files = filter(lambda x: os.stat(x).st_size >> 20 > SIZE_LIMIT, filenames)
        raise RuntimeError("Submission Failed: One or more required files is " +
                           "too large. Please make sure that the following " +
                           "files are under {}MB and try again: {!s}".format(SIZE_LIMIT, large_files))


def main(args):

    check_files(args.filenames)

    require_pledges()

    print "Submitting files..."
    submission = Submission('cs6475', args.quiz,
                            filenames=args.filenames,
                            environment=args.environment,
                            provider=args.provider)

    print "\nWaiting for results..."
    while not submission.poll():
        print "    Refresh in {} seconds...".format(REFRESH_TIME)
        time.sleep(REFRESH_TIME)
    print "    Done!"

    print "\nResults:\n--------"
    if submission.feedback():

        if 'message' in submission.feedback():
            print submission.feedback()['message']

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
    parser.add_argument('quiz', choices=FILENAMES.keys(),
                        help="Select the name of the quiz to submit")
    parser.add_argument('-f', '--filenames', nargs='+', default=[],
                        help="Names of additional files to include with submission " + \
                        "(finalproject.pdf must always be included in your submission)")
    parser.add_argument('--provider', choices=['gt', 'udacity'], default='gt',
                        help="Select the authentication system to use (default: gt)")
    parser.add_argument('--environment', default='production',
                        choices=['local', 'development', 'staging', 'production'],
                        help="Select the server to use (default: production)")
    args = parser.parse_args()
    args.filenames += FILENAMES[args.quiz]

    main(args)
