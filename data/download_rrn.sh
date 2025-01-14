# Copied from https://github.com/yilundu/ired_code_release/blob/main/data/download-rrn.sh
# Original RRN Sodoku data
wget "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1" -O sudoku-hard.zip
unzip sudoku-hard.zip
mv sudoku-hard sudoku-rrn
rm sudoku-hard.zip
rm -rf __MACOSX
