"""
This is a monkey patched version of the sphinx.apidoc utility. It provides a
modified version of the create_package_file function from apidoc.py available
at https://bitbucket.org/birkenfeld/sphinx/

It is subject to the following LICENSE:

Copyright (c) 2014, Christian Schou Oxvig & Patrick Steffen Pedersen
Copyright (c) 2007-2013 by the Sphinx team (see AUTHORS file).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

--- END OF LICENSE --

The 'AUTHORS file' is the one provided by the Sphinx project - it is available
at https://bitbucket.org/birkenfeld/sphinx/

"""

import sys
from os import path


def create_package_file(root, master_package, subroot, py_files, opts, subs):
    """Build a custom text to package files."""
    text = format_heading(1, '{0} package'.format(makename(master_package,
                                                           subroot)))

    # CHANGE: Place package description below the package headline
    text += format_directive(subroot, master_package)
    text += '\n'

    if 'magni.tests' in text:
        return

    # Build a list of directories that are subpackages (contain an INITPY file)
    subs = [sub for sub in subs if path.isfile(path.join(root, sub, INITPY))]

    if 'tests' in subs:
        subs.remove('tests')

    # If there are some package directories, add a TOC for theses subpackages
    if subs:
        text += format_heading(2, 'Subpackages')
        text += '.. toctree::\n    :maxdepth: 1\n\n'  # CHANGE: Custom maxdepth
        for sub in subs:
            text += '    {0}.{1}\n'.format(makename(master_package, subroot),
                                           sub)
        text += '\n'

    submods = [path.splitext(sub)[0] for sub in py_files
               if not shall_skip(path.join(root, sub), opts) and sub != INITPY]

    if 'run_tests' in submods:
        submods = []

    if submods:
        text += format_heading(2, 'Submodules')
        if opts.separatemodules:
            text += '.. toctree::\n    :maxdepth: 1\n\n'  # CHANGE: Maxdepth
            for submod in submods:
                modfile = makename(master_package, makename(subroot, submod))
                text += '    {0}\n'.format(modfile)

                # Generate separate file for this module
                if not opts.noheadings:
                    filetext = format_heading(1, '{0} module'.format(modfile))
                else:
                    filetext = ''
                filetext += format_directive(makename(subroot, submod),
                                             master_package)
                write_file(modfile, filetext, opts)
        else:
            for submod in submods:
                modfile = makename(master_package, makename(subroot, submod))
                if not opts.noheadings:
                    text += format_heading(2, '{0} module'.format(modfile))
                text += format_directive(makename(subroot, submod),
                                         master_package)
                text += '\n'

    write_file(makename(master_package, subroot), text, opts)
    return

if __name__ == '__main__':
    """Monkey patch the sphinx.apidoc module."""

    from sphinx.apidoc import main

    main.__globals__['create_package_file'] = create_package_file
    INITPY = main.__globals__['INITPY']
    shall_skip = main.__globals__['shall_skip']
    makename = main.__globals__['makename']
    format_heading = main.__globals__['format_heading']
    format_directive = main.__globals__['format_directive']
    write_file = main.__globals__['write_file']

    sys.exit(main(sys.argv))
