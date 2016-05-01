#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Project: rl
# File name: calculate
# Author: Mark Wang
# Date: 1/5/2016

gamma = 0.2
value = -36.56

# print 8.1 - 187.2 * gamma + 89.1 * gamma ** 2
# print 7.29 - 265.77 * gamma + 248.67 * gamma ** 2 - 80.19 * gamma ** 3
print 0.9 * -25.78 * (1 - gamma) - 90 * gamma
print 0.9 * (value) * (1- gamma) - 90 * gamma