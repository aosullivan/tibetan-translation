#!/usr/bin/env python3
"""Run BDRC's tibetan-ocr-app models over pecha folios, headless.

Three obstacles this works around, in case you re-run it elsewhere:

 1. `git lfs pull` on buda-base/tibetan-ocr-app is blocked from this sandbox, so
    Models/Lines/PhotiLines.onnx stays a 133-byte pointer and the app's own line
    detector can't load. We substitute the projection-profile segmenter in
    lineseg.py. On your Mac LFS works, so prefer the real detector there.
 2. BDRC.Data/Utils import PySide6 for type annotations; we stub Qt out rather
    than install ~200MB of GUI toolkit.
 3. pyewts fails to build against modern setuptools (`install_layout` was
    removed); we vendor the pure-Python package from the sdist instead.

The Woodblock and Modern models already have `"encoder": "wylie"` in their
model_config.json, so they emit EWTS directly — no pyewts conversion needed on
the output side.

Usage:  BDRC_MODEL=/path/to/OCRModels/Woodblock python bdrc_ocr_runner.py page*.jpg
"""

import os, sys, json, glob
import numpy as np
import cv2

sys.path.insert(0, '/tmp/tibetan-ocr-app')

# BDRC.Data imports QImage purely as a type annotation; stub Qt out so the
# inference path runs headless without pulling in ~200MB of PySide6.
import types


class _QtStub(types.ModuleType):
    """Any attribute resolves to a dummy type — BDRC.Data/Utils only use Qt
    names in annotations and GUI paths we never touch."""

    def __getattr__(self, name):
        t = type(name, (), {})
        setattr(self, name, t)
        return t


for _name in ('PySide6', 'PySide6.QtGui', 'PySide6.QtCore', 'PySide6.QtWidgets',
              'PySide6.QtSvg', 'PySide6.QtSvgWidgets'):
    sys.modules.setdefault(_name, _QtStub(_name))

from BDRC.Utils import import_local_model, get_platform
from BDRC.Inference import OCRInference

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lineseg import load_binary, strip_frame, find_lines

MODEL_DIR = os.environ.get('BDRC_MODEL', '/tmp/OCRModels/Woodblock')


def build():
    m = import_local_model(MODEL_DIR)
    if m is None:
        raise SystemExit(f'no model at {MODEL_DIR}')
    cfg = m.config
    cfg.model_file = os.path.join(MODEL_DIR, os.path.basename(cfg.model_file))
    return OCRInference(get_platform(), cfg)


def page_lines(path, pad=10):
    im, mask = load_binary(path)
    y0, y1, x0, x1 = strip_frame(mask, im)
    bgr = cv2.imread(path)
    out = []
    for ys, ye in find_lines(mask, y0, y1, x0, x1):
        crop = bgr[max(0, ys - pad):min(bgr.shape[0], ye + pad), x0:x1]
        if crop.size:
            out.append(crop)
    return out


if __name__ == '__main__':
    infer = build()
    for p in sorted(sys.argv[1:]):
        print(f"### {os.path.basename(p)}")
        for i, line in enumerate(page_lines(p), 1):
            try:
                print(f"{i}| {infer.run(line)}")
            except Exception as e:
                print(f"{i}| [FAIL {type(e).__name__}: {e}]")
        print()
