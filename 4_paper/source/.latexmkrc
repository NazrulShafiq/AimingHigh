# latexmk settings for the paper. latexmk reads this file when run from this folder
# (4_paper/source):
#   4_paper/source/Writeup.tex   ->   4_paper/output/Writeup.pdf
# with the auxiliary files in 4_paper/output/aux/Writeup/. Each document has its own aux
# folder: Writeup.tex and Writeup_format.tex both \include Appendix.tex, and a shared
# Appendix.aux would force each to recompile after the other.

$pdf_mode = 1;            # pdflatex
$out_dir = '../output';
my ($main) = grep { /\.tex$/ } @ARGV;
$aux_dir = '../output/aux/' . (defined $main ? $main =~ s/\.tex$//r : 'Writeup');
