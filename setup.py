from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="brainbraille_decode",
    version="1.0.0",
    author="Yuhui Zhao",
    author_email="yhzhao343@gmail.com",
    url='https://github.com/yhzhao343/brainbraille_decode',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    zip_safe=False,
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "scikit-learn",
        "fastFMRI @ git+https://github.com/yhzhao343/fastFMRI.git",
        "lz4",
        "psutil",
        "joblib",
        "dlib",
        "lipo",
    ],
)
