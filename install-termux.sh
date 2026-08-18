#!/data/data/com.termux/files/usr/bin/bash
# cardcenter -- Android install via Termux.
#
# Termux from F-Droid, NOT the Play Store version (that one is abandoned and
# its package repos are frozen). https://f-droid.org/packages/com.termux/
#
# Run:  bash install-termux.sh
set -e

say() { printf '\n\033[1;36m==\033[0m %s\n' "$1"; }

say "Updating package lists"
pkg update -y && pkg upgrade -y

say "Installing Python and the build essentials"
# Termux ships prebuilt wheels for numpy and opencv; building either from
# source on a phone takes hours and usually fails on memory.
pkg install -y python python-numpy python-pillow libjpeg-turbo libpng git

say "Installing OpenCV"
# The Termux package is a native build. pip install opencv-python does NOT work
# on Android -- there is no aarch64 wheel on PyPI and the source build needs a
# full CMake/FFmpeg toolchain.
pkg install -y opencv-python || pkg install -y python-opencv || {
  echo "OpenCV package not found under either name."
  echo "Try:  pkg search opencv"
  exit 1
}

say "Installing Tesseract (optional -- only needed for collector numbers)"
pkg install -y tesseract || echo "  skipped; centering works without it"

say "Installing cardcenter"
pip install --no-deps -e .

say "Checking it works"
python -c "
import cv2, numpy
from cardcenter import measure_centering, __version__
print('  cardcenter', __version__)
print('  opencv', cv2.__version__, '| numpy', numpy.__version__)
import shutil
print('  tesseract:', 'yes' if shutil.which('tesseract') else 'no (centering still works)')
"

cat <<'DONE'

  Installed.

  Start it with:

      cardcenter --serve

  then open http://127.0.0.1:8765 in Chrome on this phone.

  Tip: keep Termux from being killed in the background with

      termux-wake-lock

DONE
