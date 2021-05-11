#!/bin/bash

# install libpng needed by opencv
ldconfig -p | grep png16 > log 
if [ $? -ne 0 ];then
    apt-get install libpng16-16
fi

# install libjasper1 needed by opencv
ldconfig -p | grep libjasper  > log 
if [  $? -ne 0  ];then
    add-apt-repository > log
    if [  $? -ne 0 ]; then
        apt-get install software-properties-common
    fi
    add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
    apt update
    apt install libjasper1 libjasper-dev
fi

rm -rf log

