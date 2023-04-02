# Face Recognition From Scratch with Principal Component Analysis

Principal Component Analysis (PCA) is a statistical method that can be used for face recognition. However, it is more or less outdated, and current face recognition models normally use convolutional neural networks (CNN) to achieve higher accuracy.

This project aims to use PCA to build a working face recognition system from scratch, and compare its performance with state of the art CNNs.

## How it works

A big problem with face recognition is the large number of variables in an image. For example, a 64x64 image will already consist of 4096 variables (pixels). Checking and comparing each pixel is extremely inefficient and easily affected by changes in lighting or background.

PCA tackles this problem by reducing the number of variables in an image. Using a technique called eigendecomposition, PCA breaks down each face into several different components, called eigenfaces. The idea is that every person's face is a weighted combination of these "eigenfaces".

