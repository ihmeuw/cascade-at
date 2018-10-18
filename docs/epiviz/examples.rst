Examples Calling Dmcascade
==========================

Command Line
------------
Run the cascade against the latest model version associated with an meid::

    dmcascade --db-file-path 1989.db --meid 1989

Create a JSON settings file from a model. First run EpiViz.
Then go to the Developer Console. This is a menu option in the web browser.
And type ``printSettingsJson()``. Then copy what it prints to a file.

Call from a JSON settings file::

    dmcascade --settings-file 1989.json --db-file-path 1989.db

