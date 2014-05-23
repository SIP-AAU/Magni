"""
..
    Copyright (c) 2014, Magni developers.
    All rights reserved.
    See LICENSE.rst for further information.

Script for generating html frontpage for Magni pdf documentation.

"""

import sys
import os
import datetime

sys.path.insert(0, os.path.abspath('../'))

import magni


source_dir, name, ext = magni.utils.split_path(str(os.path.abspath(__file__)))
now = datetime.datetime.now()

with open(source_dir + 'cover.html', 'w') as cover_file:
    cover_file.write("""
<!DOCTYPE html>
<html>
  <head>
    <title>Magni PDF Cover</title>
  </head>
  <body>
    <br style="height=72pt;" />
    <hr color=#000000>
    <h1 style="margin-top: 24pt;" align=Right>Magni Documentation</h1>
    <h2 style="margin-top: 12pt;" align=Right>Version {ver}</h2>
    <p style="margin-top: 192pt;" align=center>
      <b>Christian Schou Oxvig and Patrick Steffen Pedersen</b>
      <br/>
      <b>in collaboration with</b>
      <br/>
      <b>Jan &Oslash;stergaard, Thomas Arildsen,</b>
      <br/>
      <b>Tobias L. Jensen, and Torben Larsen</b>
    </p>
    <p style="margin-top: 288pt;" align=right>
      <i>{month} {day}, {year}</i>
    </p>
  </body>
</html>
""".format(ver=magni.__version__, month=now.strftime('%B'),
           day=now.strftime('%d'), year=now.strftime('%Y')))
