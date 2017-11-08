"""
Download all abc files on a page given page URL.

USAGE
=====
python download_midi_files.py https://example.com PREFIX

where PREFIX is the prefix to download URLs. Typically links on a page either
specify a full URL (e.g. https://example.com/public/file.mid) or a path relative
to the page (e.g. /file.mid, or /public/file.mid). In that case you need to
provide the correct PREFIX so the script knows what the correct full URL is. 

E.g. assuming the full URL for the file is https://example.com/public/file.mid,
and the links on the page look like /file.mid, PREFIX will need to be 
https://example.com/public. 
"""

import os
import sys

# Python 2 import
# from urllib2 import Request, urlopen, URLError, HTTPError
# from BeautifulSoup import BeautifulSoup, SoupStrainer

# Python 3 import
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from bs4 import BeautifulSoup, SoupStrainer

def dlfile(url):
    try:
        f = urlopen(url)
        with open(os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())
    except HTTPError as e:
        print("HTTPError:", e.code, url)
    except URLError as e:
        print("URLError:", e.reason, url)

URL = sys.argv[1]
DOWNLOAD_LINK_PREFIX = sys.argv[2]
req = Request(URL)
response = urlopen(req)
soup = BeautifulSoup(response)

for a in soup.findAll('a', href=True):
    url = a["href"]    
    if url.endswith(".mid"):
        # Toggle this to test what the links look like before you download
        # print("Downloading", DOWNLOAD_LINK_PREFIX + url)

        full_url = DOWNLOAD_LINK_PREFIX + url
        print("Downloading", full_url)
        dlfile(full_url)