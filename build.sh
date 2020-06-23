rm -rf build/
rm -rf dist/
rm -rf segmentation_evaluator.egg-info/
python setup.py sdist bdist_wheel
