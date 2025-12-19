from setuptools import setup, Extension

modpModule = Extension(
    "_modp",
    sources=["modp.i", "modp.c"],
)

setup(
    name="_modp",
    ext_modules=[modpModule],
    py_modules=["modp"]
)
