# README #

Resources for the paper "Exploring character shape for non-supervised character segmentation" submitted to IEEE Transactions on Information Forensics and Security.

## Downloading (it includes the dataset)

1. git clone https://thiagopx@bitbucket.org/thiagopx/tifs_2017.git

## Enviroment setup (tested on linux Ubuntu/Amazon Linux AMI)

### U

1. sudo yum/apt-get update -y %% sudo yum/apt-get install dockerDownload docker Anconda Python 2.7 from https://www.anaconda.com/download
2. bash Anaconda2-<version>.sh

### Dependencies

* Scikit-image

1. conda install scikit-image

* ConfigObj

1. conda install configobj

* OpenCV

1. pip install opencv-python

* Pyenchant

1. sudo apt-get install myspell-pt-br
2. pip install pyenchant

* Tesseract

1. sudo apt-get install tesseract-ocr libtesseract-dev libleptonica-dev
2. sudo pip install pytesseract
3. wget https://sourceforge.net/projects/tesseract-ocr-alt/files/por.traineddata.gz/download
4. gunzip por.traineddata.gz
5. sudo cp por.traineddata /usr/share/tesseract-ocr/tessdata


* Concorde

1. Download the QSopt Beta 1.1 files from http://www.math.uwaterloo.ca/~bico/qsopt/beta/index.html (all the files listed under Red Hat Linux, gcc 3.4.3 - AMD 64-bit)
2. Download Concorde-03.12.19 from http://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm
3. tar xf co031219.tgz
4. cd concorde
5. ./configure --with-qsopt=<full path to qsopt files>
6. make
7. export PATH=<full path to concorde>/TSP:$PATH

* Boost.Python

1. $ wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.zip
2. $ unzip boost_1_64_0.zip
3. $ cd boost_1_64_0
4. $ ./bootsrap.sh
5. $ bjam --with-python
