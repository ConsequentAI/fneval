main.pdf: main.tex refs.bib
	pdflatex main
	bibtex main
	pdflatex main
	bibtex main
	pdflatex main

clean:
	rm main.pdf main.aux main.log main.bbl main.blg
	rm missfont.log
	rm texput.log
