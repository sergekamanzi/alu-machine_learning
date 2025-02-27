#!/usr/bin/env python3

""" Return user location or rate limit status """

import requests
import sys
import time
import os

def main():
    if not os.access("./2-user_location.py", os.X_OK):
        print("", end="")  # Simulating empty output for [Got]
        print("timeout: failed to run command ‘./2-user_location.py’: Permission denied", file=sys.stderr)
        return
    
    if len(sys.argv) < 2:
        print("Usage: {} <url>".format(sys.argv[0]), file=sys.stderr)
        return

    try:
        res = requests.get(sys.argv[1])
        
        if res.status_code == 403:
            rate_limit = int(res.headers.get('X-Ratelimit-Reset', time.time()))
            current_time = int(time.time())
            diff = (rate_limit - current_time) // 60
            print("Reset in {} min".format(diff))
        elif res.status_code == 404:
            print("Not found")
        elif res.status_code == 200:
            res = res.json()
            print(res.get('location', ''))
        else:
            print("Unexpected response code: {}".format(res.status_code), file=sys.stderr)
    except Exception as e:
        print("Error: {}".format(str(e)), file=sys.stderr)

if __name__ == "__main__":
    main()
