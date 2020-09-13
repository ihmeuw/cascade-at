# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = DismodAT
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile png

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: png

png: dev/cascade_components.png dev/cascade_container_diagram.png dev/cascade_system_diagram.png


dev/cascade_components.png:
	convert -density 300 -depth 8 -background white dev/cascade_components.pdf -resize 25% -trim dev/cascade_components.png

dev/cascade_container_diagram.png:
	convert -density 300 -depth 8 -background white dev/cascade_container_diagram.pdf -resize 25% -trim dev/cascade_container_diagram.png

dev/cascade_system_diagram.png:
	convert -density 300 -depth 8 -background white dev/cascade_system_diagram.pdf -resize 25% -trim dev/cascade_system_diagram.png
