from setuptools import setup, find_packages

setup(
    name='object_detection_app',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'streamlit',
        'torch',
        'transformers',
        'Pillow'
    ],
    entry_points={
        'console_scripts': [
            'object_detection_app = app:main'
        ]
    }
)
