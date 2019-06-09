#!/bin/bash -eux


sudo docker run -it --rm --runtime nvidia -v ${PWD}:/home/app --name nnabla-drl-collections takuseno/nnabla-drl-collections bash
