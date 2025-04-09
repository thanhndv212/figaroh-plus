# Copyright [2021-2025] Thanh Nguyen
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

import meshcat


def rgb2int(r, g, b):
    """Convert 3 integers (chars) 0<= r,g,b < 256 into one single integer
    = 256**2*r+256*g+b, as expected by Meshcat."""
    return int(r * 256**2 + g * 256 + b)


def material(color, transparent=False):
    mat = meshcat.geometry.Material()
    mat.color = color
    mat.transparent = transparent
    return mat


red = material(color=rgb2int(255, 0, 0), transparent=False)
blue = material(color=rgb2int(0, 0, 255), transparent=False)
green = material(color=rgb2int(0, 255, 0), transparent=False)
yellow = material(color=rgb2int(255, 255, 0), transparent=False)
magenta = material(color=rgb2int(255, 0, 255), transparent=False)
cyan = material(color=rgb2int(0, 255, 255), transparent=False)
white = material(color=rgb2int(250, 250, 250), transparent=False)
black = material(color=rgb2int(5, 5, 5), transparent=False)
grey = material(color=rgb2int(120, 120, 120), transparent=False)


colormap = {
    "red": red,
    "blue": blue,
    "green": green,
    "yellow": yellow,
    "magenta": magenta,
    "cyan": cyan,
    "black": black,
    "white": white,
    "grey": grey,
}
