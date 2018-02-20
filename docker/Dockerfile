FROM ubuntu:17.10
MAINTAINER Double Blinded

RUN apt-get update && apt-get install -y build-essential curl

# Home and project paths in the VM
ENV HOME /root
ENV PROJECTDIR $HOME/tifs

# Anaconda
RUN curl -o installer.sh https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh && \
    bash /installer.sh -b -f && \
    rm /installer.sh

ENV PATH $HOME/anaconda2/bin:$PATH

# Scikit-image
RUN conda install scikit-image

# ConfigObj
RUN conda install configobj

# Pyenchant
RUN apt-get install -y myspell-pt-br enchant && \
    pip install pyenchant 

# Concorde
ENV QSOPTDIR $HOME/qsopt
ENV CONCORDEDIR $HOME/concorde

RUN mkdir -p $QSOPTDIR && \
    curl -o $QSOPTDIR/qsopt.a http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.PIC.a && \
    curl -o $QSOPTDIR/qsopt.h http://www.math.uwaterloo.ca/~bico/qsopt/beta/codes/PIC/qsopt.h && \
    curl -o concorde.tgz http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz  && \
    tar -xf concorde.tgz -C $HOME && \
    rm -rf concorde.tgz && \
    cd $CONCORDEDIR && \
    ./configure --with-qsopt=$QSOPTDIR && \
    make

ENV PATH "$CONCORDEDIR/TSP:$PATH"

# Tesseract
ENV TESSERACTDIR /usr/share/tesseract-ocr

RUN apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev wget && \
    wget https://sourceforge.net/projects/tesseract-ocr-alt/files/por.traineddata.gz && \
    gunzip por.traineddata.gz && \
    mv por.traineddata $TESSERACTDIR/tessdata

# OpenCV
RUN apt-get install -y libsm6 && \
    pip install opencv-python

# Other dependencies
RUN apt-get install -y libgl1-mesa-glx qt4-dev-tools libgmp-dev

ENV LD_LIBRARY_PATH "$PROJECTDIR/docrec/libs:$LD_LIBRARY_PATH"
ENV PYTHONPATH "$PROJECTDIR/docrec/libs:$PYTHONPATH"

CMD cd $PROJECTDIR; bash
