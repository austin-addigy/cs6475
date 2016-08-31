import time
import os
import sys
import argparse
import json
import datetime
import glob
from bonnie.submission import Submission

IMG_EXTS = ["png", "jpeg", "jpg", "gif", "tiff", "tif", "raw", "bmp"]

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
    raise RuntimeError("Late policy not accepted.")

  print
  print(HONOR_PLEDGE)
  ans = raw_input("Please type 'yes' to agree and continue>")
  if ans != "yes":
    raise RuntimeError("Honor pledge not accepted")
  print

def main():
  parser = argparse.ArgumentParser(description='Submits code to the Udacity site.')
  parser.add_argument('--provider', choices = ['gt', 'udacity'], default = 'gt')
  parser.add_argument('--environment', choices = ['local', 'development', 'staging', 'production'], default = 'production')
  args = parser.parse_args()

  quiz = 'writeup1'

  require_pledges()

  filelist = glob.glob('image1.*')
  if len(filelist) != 1:
    raise RuntimeError("There can only be one image file named image1.<EXT>")
  elif os.path.splitext(filelist[0])[-1][1:] not in IMG_EXTS:
    raise RuntimeError("The image file extension must be one of the following types: {!s}".format(IMG_EXTS))
  files = ['assignment1.pdf', filelist[0]]


  print "Submission processing...\n"
  submission = Submission('cs6475', quiz,
                          filenames = files,
                          environment = args.environment,
                          provider = args.provider)

  timestamp = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now())

  while not submission.poll():
    time.sleep(3.0)

  if submission.feedback():

    if submission.console():
        sys.stdout.write(submission.console())

    filename = "%s-result-%s.json" % (quiz, timestamp)

    with open(filename, "w") as fd:
      json.dump(submission.feedback(), fd, indent=4, separators=(',', ': '))

    print("\n(Details available in %s.)" % filename)

  elif submission.error_report():
    error_report = submission.error_report()
    print(json.dumps(error_report, indent=4))
  else:
    print("Unknown error.")

if __name__ == '__main__':
  main()

