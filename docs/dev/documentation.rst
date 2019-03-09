.. _how-to-document:

How to Document
===============

Section headings must be consistent across all documentation.
We use:

 *  # with overline, for parts, which means there is one
    of these for the User manual, Developer manual, etc.
 *  \* with overline, for chapters
 *  =, for sections
 *  -, for subsections
 *  ^, for subsubsections
 *  ", for paragraphs

Within the code, function documentation uses the
Google Python Docstring style, as described in
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html.

Instead of linking directly to pages, create anchors
so that they work when pages move. Make an anchor for every header, like so:

::

    .. _exception-handling-proposal:

    Exception-Handling Proposal
    ---------------------------



The two main references on how to write documentation
using reStructuredText are
http://www.sphinx-doc.org/en/master/usage/restructuredtext/
and
http://docutils.sourceforge.net/rst.html. Read all of
the first reference in order to understand roles and linking.

