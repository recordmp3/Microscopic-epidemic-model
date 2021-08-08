
from setuptools import setup
from Cython.Build import cythonize

from distutils.extension import Extension
from Cython.Distutils import build_ext
#setup(ext_modules=cythonize("env2.pyx", compiler_directives={'language_level' : "3"}, extra_compile_args=["-std=c++11"]))
#setup(ext_modules=cythonize("env2.pyx", compiler_directives={'language_level' : "3"}))

setup(
  name = 'Test app',
  ext_modules=[
    Extension('env2',
              sources=['env2.pyx'],
              extra_compile_args=['-std=c++11'],
              language='c++')
    ],
  cmdclass = {'build_ext': build_ext}
)
