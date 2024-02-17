# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install submodule calibration
set(FILES
  __init__.py
  calibration_tools.py
  )
foreach(F ${FILES})
    python_install_on_site("figaroh/calibration" ${F})
endforeach()

# Install submodule identification
set(FILES
  __init__.py
  identification_tools.py
  )
foreach(F ${FILES})
    python_install_on_site("figaroh/identification" ${F})
endforeach()

# Install submodule measurements
set(FILES
  __init__.py
  measurement.py
  )
foreach(F ${FILES})
    python_install_on_site("figaroh/measurements" ${F})
endforeach()

# Install submodule meshcat_viewer_wrapper
set(FILES
  __init__.py
  colors.py
  visualizer.py
  )
foreach(F ${FILES})
    python_install_on_site("figaroh/meshcat_viewer_wrapper" ${F})
endforeach()

# Install submodule tools
set(FILES
  __init__.py
  qrdecomposition.py
  randomdata.py
  regressor.py
  robot.py
  robotcollisions.py
  robotipopt.py
  robotvisualization.py
  )
foreach(F ${FILES})
    python_install_on_site("figaroh/tools" ${F})
endforeach()

# Install submodule utils
set(FILES
  __init__.py
  )
foreach(F ${FILES})
    python_install_on_site("figaroh/utils" ${F})
endforeach()
