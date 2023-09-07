"""The deploy util."""

import argparse
parser = argparse.ArgumentParser(description='Parameters of util.')
parser.add_argument('--trigger', type=str, default="", help='trigger name')
parser.add_argument('--mode', type=str, default="", help='mode: setup|test|load_content|release')
parser.add_argument('--content', type=str, default="", help='content of trigger')
args = parser.parse_args()

import logging
import sys
from pants import object_trigger
ot = object_trigger.ObjectTrigger()


if __name__ == '__main__':
    logging.basicConfig()
    if len(args.trigger) < 0:
        sys.exit(1)
    if args.mode not in ['setup', 'test', 'load_content', 'release']:
        sys.exit(1)
    if args.mode == 'setup':
        ot.setup_trigger(args.trigger, args.content)
        sys.exit(0)
    if args.mode == 'test':
        result = ot.test_trigger(args.trigger)
        if result:
            print("exists")
        else:
            print("not")
        sys.exit(0)
    if args.mode == 'load_content':
        print(ot.load_trigger_content(args.trigger))
        sys.exit(0)
    if args.mode == 'release':
        ot.release_trigger(args.trigger)
        sys.exit(0)
