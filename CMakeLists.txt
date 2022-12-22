# Copyright 2022 CNRS
# Author: Florent Lamiraux
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

set(CXX_DISABLE_WERROR TRUE)

set(PROJECT_NAME figaroh)
set(PROJECT_DESCRIPTION "Calibration and identification of robots and humans")
set(PROJECT_URL "")
set(PROJECT_USE_CMAKE_EXPORT FALSE)

include(cmake/base.cmake)
include(cmake/test.cmake)
include(cmake/python.cmake)

findpython(REQUIRED)

## Dependencies

# add_project_dependency(pinocchio 2.5 REQUIRED)
# numdifftools
# numpy
# scipy

set(PYTHON_COMPONENTS Interpreter NumPy)

compute_project_args(PROJECT_ARGS LANGUAGES CXX)
MESSAGE(STATUS "PROJECT_ARGS=${PROJECT_ARGS}")
project(${PROJECT_NAME} ${PROJECT_ARGS})

add_subdirectory(src)

setup_project_finalize()


