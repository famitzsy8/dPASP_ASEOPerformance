# Research Project: Analysis of ASEO Approximation Algorithms for PASP Likelihood Inference

This technical report is a project made for the course "Atividade Curricular em Pesquisa" (MAC0215) at the Mathematics and Statistics Institute of Universidade de São Paulo (IME-USP). It serves mainly to shed a better light on the performance of an approximation algorithm implemented into the [dPASP framework](https://github.com/kamel-usp/dpasp), developed at the [Knowledge-Augmented Machine Learning Research Group (KAMEL-USP)](https://github.com/kamel-usp). The project was supervised by [Denis Maratani Mauá](https://www.ime.usp.br/ddm/) and put into practice by no-one less than myself, Yuri Simantob.


## Abstract

Until now, the inference of queries to probabilistic logic programs (PLP) has been limited to small-scale examples due to the brute-force nature of the infer- ence algorithms, enumerating all potential solutions. Thus, the use of Answer Set Enumeration by Optimality (ASEO) algorithms for approximate inference was experimented with, and an initial version implemented into the dPASP framework. This report contains the results of experiments analyzing differ- ences in performance between stratified and non-stratified programs. While no notable differences could be found, large variations in the convergence dynam- ics were witnessed, affecting the predictability and usability of an approximate inference algorithm in dPASP.

- [Read the Paper](./docs/paper.pdf)
