from setuptools import setup, find_packages


setup(name='ICE',
      version='1.0.0',
      description='ICE: Inter-instance Contrastive Encoding for Unsupervised Person Re-identification',
      author='Hao Chen',
      author_email='hao.chen@inria.fr',
      url='https://github.com/chenhao2345/ICE',
      install_requires=[
          'numpy', 'torch==1.7.0', 'torchvision==0.8.0',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss-gpu'],
      packages=find_packages(),
      keywords=[
          'Contrastive Learning',
          'Person Re-identification'
      ])

