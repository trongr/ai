"""
Download all abc files on a page given page URL.
python download_abc_files.py https://example.com
"""

import os
import sys
import urllib2
from urllib2 import Request, urlopen, URLError, HTTPError
from BeautifulSoup import BeautifulSoup, SoupStrainer

def dlfile(url):
    try:
        f = urlopen(url)
        with open(os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())
    except HTTPError, e:
        print "HTTPError:", e.code, url
    except URLError, e:
        print "URLError:", e.reason, url

URL = sys.argv[1]
request = Request(URL)
response = urlopen(request)
soup = BeautifulSoup(response)

for a in soup.findAll('a', href=True):
    abc_url = a["href"]    
    if abc_url.endswith(".abc"):
        abc_full_url = URL + abc_url
        print("Downloading", abc_full_url)
        dlfile(abc_full_url)