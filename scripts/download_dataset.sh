#!/bin/bash

mkdir -p ./data/slr80/wavs
cd ./data/slr80

# Download TSV manifest
wget https://www.openslr.org/resources/80/line_index_female.tsv

# Download audio (948 MB)
wget https://www.openslr.org/resources/80/my_mm_female.zip
unzip my_mm_female.zip
# Audio files land in  ../data/slr80/wavs/  automatically