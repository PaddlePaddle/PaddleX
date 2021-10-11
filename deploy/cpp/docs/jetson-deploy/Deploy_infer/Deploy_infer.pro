#-------------------------------------------------
#
# Project created by QtCreator 2021-10-08T16:09:04
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Deploy_infer
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
    inferthread.cpp

HEADERS += \
        mainwindow.h \
    inferthread.h

FORMS += \
        mainwindow.ui

INCLUDEPATH += /usr/include/opencv4 \
               /usr/include/opencv4/opencv \
               /usr/include/opencv4/opencv2

LIBS += /usr/lib/libopencv_*.so


