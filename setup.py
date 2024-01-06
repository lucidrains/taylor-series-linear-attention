from setuptools import setup, find_packages

setup(
  name = 'taylor-series-linear-attention',
  packages = find_packages(exclude=[]),
  version = '0.0.8',
  license='MIT',
  description = 'Taylor Series Linear Attention',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/taylor-series-linear-attention',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention mechanism'
  ],
  install_requires=[
    'einops>=0.7.0',
    'rotary-embedding-torch>=0.5.3',
    'torch>=2.0',
    'torchtyping'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
